#!/usr/bin/env python3
"""
CHUK LLM Simple Functions + Session Manager Integration

This example shows how to use CHUK LLM's simple convenience functions
while still tracking everything in sessions.
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, Optional

from dotenv import load_dotenv

# CHUK LLM simple functions (v0.19+ unified API)
from chuk_llm import ask, ask_sync

# Session manager imports - FIXED for current architecture
from chuk_ai_session_manager.models.event_source import EventSource
from chuk_ai_session_manager.models.event_type import EventType
from chuk_ai_session_manager.models.session import Session
from chuk_ai_session_manager.models.session_event import SessionEvent
from chuk_ai_session_manager.session_storage import (
    ChukSessionsStore,
    get_backend,
    setup_chuk_sessions_storage,
)

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Quiet down ALL the noisy loggers
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("requests").setLevel(logging.ERROR)
logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("anthropic").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)
logging.getLogger("h11").setLevel(logging.ERROR)
logging.getLogger("chuk_llm").setLevel(logging.WARNING)
logging.getLogger("chuk_sessions").setLevel(logging.WARNING)
logging.getLogger("chuk_ai_session_manager").setLevel(logging.WARNING)


class TrackedLLM:
    """
    A wrapper that adds session tracking to CHUK LLM simple functions.

    This approach is simpler than the conversation API but still provides
    complete session tracking capabilities.
    """

    def __init__(self, session: Session):
        self.session = session

    async def _track_interaction(
        self,
        user_message: str,
        response: str,
        provider: str,
        model: str,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Track a complete LLM interaction in the session."""
        base_metadata = {
            "timestamp": datetime.now().isoformat(),
            "provider": provider,
            "model": model,
        }
        if metadata:
            base_metadata.update(metadata)

        # Track user message with token counting
        user_event = await SessionEvent.create_with_tokens(
            message=user_message,
            prompt=user_message,
            model=model,
            source=EventSource.USER,
            type=EventType.MESSAGE,
        )
        # Add metadata
        for key, value in base_metadata.items():
            await user_event.set_metadata(key, value)

        await self.session.add_event_and_save(user_event)

        # Track assistant response with token counting
        assistant_event = await SessionEvent.create_with_tokens(
            message=response,
            prompt="",  # No prompt for response
            completion=response,
            model=model,
            source=EventSource.LLM,
            type=EventType.MESSAGE,
        )
        # Add metadata
        for key, value in base_metadata.items():
            await assistant_event.set_metadata(key, value)

        await self.session.add_event_and_save(assistant_event)

        logger.debug(f"💾 Tracked {provider}/{model} interaction")

    # OpenAI functions with tracking
    async def ask_openai_gpt4o_mini(self, question: str, **kwargs) -> str:
        """Ask OpenAI GPT-4o Mini with session tracking."""
        response = await ask(question, provider="openai", model="gpt-4o-mini", **kwargs)
        await self._track_interaction(question, response, "openai", "gpt-4o-mini")
        return response

    # Anthropic functions with tracking
    async def ask_anthropic_sonnet(self, question: str, **kwargs) -> str:
        """Ask Anthropic Claude Sonnet with session tracking."""
        response = await ask(question, provider="anthropic", model="claude-sonnet-4-5", **kwargs)
        await self._track_interaction(
            question, response, "anthropic", "claude-sonnet-4-5"
        )
        return response

    async def ask_anthropic_claude_sonnet4(self, question: str, **kwargs) -> str:
        """Ask Anthropic Claude Sonnet 4 with session tracking."""
        response = await ask(question, provider="anthropic", model="claude-sonnet-4-5", **kwargs)
        await self._track_interaction(
            question, response, "anthropic", "claude-sonnet-4-5"
        )
        return response

    # Multi-provider convenience method
    async def ask_parallel(
        self, question: str, providers: list = None
    ) -> Dict[str, str]:
        """Ask multiple providers in parallel and track all responses."""
        if providers is None:
            providers = ["openai", "anthropic"]

        tasks = []
        for provider in providers:
            if provider == "openai":
                tasks.append(("openai", self.ask_openai_gpt4o_mini(question)))
            elif provider == "anthropic":
                tasks.append(("anthropic", self.ask_anthropic_sonnet(question)))

        # Execute in parallel
        results = {}
        for provider, task in tasks:
            try:
                results[provider] = await task
            except Exception as e:
                logger.error(f"❌ Error with {provider}: {e}")
                results[provider] = f"Error: {str(e)}"

        return results

    # Sync function wrapper (for compatibility)
    def ask_sync_tracked(self, question: str, provider: str = "openai") -> str:
        """Synchronous ask with tracking (creates new event loop if needed)."""
        response = ask_sync(question, provider=provider)

        # Track in background (requires running event loop)
        asyncio.create_task(
            self._track_interaction(question, response, provider, "default")
        )
        return response


async def demonstrate_simple_tracking():
    """Demonstrate session tracking with simple functions."""
    print("🚀 CHUK LLM Simple Functions + Session Tracking")
    print("=" * 60)

    # Setup CHUK Sessions backend
    setup_chuk_sessions_storage(sandbox_id="chuk-llm-simple-demo", default_ttl_hours=1)

    # Create session with metadata
    session = await Session.create()
    await session.metadata.set_property("example", "simple_functions_tracking")
    await session.metadata.set_property(
        "description", "Using CHUK LLM simple functions with session tracking"
    )

    # Save session with metadata
    backend = get_backend()
    store = ChukSessionsStore(backend)
    await store.save(session)

    print(f"📝 Created session: {session.id}")

    # Create tracked LLM wrapper
    llm = TrackedLLM(session)

    print("\n💬 Single Provider Interactions:")
    print("-" * 40)

    # OpenAI interaction
    print("🤖 Asking OpenAI GPT-4o Mini...")
    response = await llm.ask_openai_gpt4o_mini("What's 2+2? Explain briefly.")
    print(f"Response: {response}")

    # Anthropic interaction
    print("\n🧠 Asking Anthropic Claude Sonnet...")
    response = await llm.ask_anthropic_sonnet("What's 3+3? Be concise.")
    print(f"Response: {response}")

    print("\n⚡ Parallel Provider Interactions:")
    print("-" * 40)

    # Parallel interactions
    print("🔄 Asking both providers simultaneously...")
    results = await llm.ask_parallel(
        "What's the capital of France? One word answer.",
        providers=["openai", "anthropic"],
    )

    for provider, response in results.items():
        print(f"{provider.capitalize()}: {response}")

    # Get fresh session data with all events
    try:
        fresh_session = await store.get(session.id)
        if fresh_session:
            session = fresh_session
    except Exception as e:
        logger.warning(f"Could not refresh session: {e}")

    # Show session tracking results
    print("\n📊 Session Tracking Results:")
    print("-" * 40)

    print(f"📋 Session ID: {session.id}")
    print(f"📅 Created: {session.metadata.created_at}")
    print(f"🔄 Updated: {session.metadata.updated_at}")
    print(f"📝 Total Events: {len(session.events)}")
    print(f"🎯 Total Tokens: {session.total_tokens}")
    print(f"💰 Estimated Cost: ${session.total_cost:.6f}")

    print("\n🗃️ Event History:")
    for i, event in enumerate(session.events, 1):
        timestamp = await event.get_metadata("timestamp", "Unknown")
        provider = await event.get_metadata("provider", "Unknown")
        model = await event.get_metadata("model", "Unknown")

        print(f"  {i}. [{event.source.value}] {provider}/{model}")
        content = str(event.message)
        if len(content) > 60:
            content = content[:57] + "..."
        print(f"     Content: {content}")
        print(f"     Time: {timestamp}")
        if event.token_usage:
            print(f"     Tokens: {event.token_usage.total_tokens}")
        print()


async def demonstrate_conversation_vs_simple():
    """Compare conversation API vs simple functions approach."""
    print("\n🔄 Conversation API vs Simple Functions Comparison")
    print("=" * 60)

    # Simple functions approach
    print("📌 Simple Functions (Stateless):")
    print("✅ Pros: Easy to use, no context management, great for one-offs")
    print("🔶 Trade-offs: No conversation memory, each call is independent")

    response1 = await ask("My name is Alice", provider="openai", model="gpt-4o-mini")
    print(f"Call 1: {response1}")

    response2 = await ask("What's my name?", provider="openai", model="gpt-4o-mini")
    print(f"Call 2: {response2}")  # Won't remember Alice

    print("\n📌 With Session Tracking (Stateful-like):")
    print("✅ Pros: Still simple functions but with complete tracking")
    print("📊 Benefit: Full observability and session management")

    try:
        # Create a new session for this demo
        demo_session = await Session.create()
        await demo_session.metadata.set_property("demo", "conversation_comparison")

        tracked_llm = TrackedLLM(demo_session)

        # Same calls but now tracked
        response1 = await tracked_llm.ask_openai_gpt4o_mini("My name is Alice")
        print(f"Tracked Call 1: {response1}")

        response2 = await tracked_llm.ask_openai_gpt4o_mini("What's my name?")
        print(
            f"Tracked Call 2: {response2}"
        )  # Still won't remember Alice, but now tracked!

        print(f"\n📈 Session now has {len(demo_session.events)} tracked events")

    except Exception as e:
        print(f"⚠️ Tracking demo failed: {e}")
        logger.exception("Error in tracking demo")

    print("\n💡 Advanced: Context Simulation with Simple Functions:")
    print("📝 You can simulate conversation context by including history:")

    # Demonstrate context simulation
    context_prompt = """Previous conversation:
User: My name is Alice
Assistant: Nice to meet you, Alice!

Current question: What's my name?"""

    response = await ask(context_prompt, provider="openai", model="gpt-4o-mini")
    print(f"Context-aware response: {response}")


async def main():
    """Main demonstration."""
    try:
        await demonstrate_simple_tracking()
        await demonstrate_conversation_vs_simple()

        print("\n✅ DEMONSTRATION COMPLETE")
        print("=" * 60)
        print("🎯 Key Takeaways:")
        print("  • Simple functions: Great for stateless interactions")
        print("  • Session tracking: Adds observability without complexity")
        print("  • Token & cost tracking: Built-in with create_with_tokens")
        print("  • Parallel requests: Easy with simple functions")
        print("  • Full audit trail for all interactions")
        print("  • Choose based on your use case!")

    except Exception as e:
        logger.error(f"❌ Demo failed: {e}")
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
