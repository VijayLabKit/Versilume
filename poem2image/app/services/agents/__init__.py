"""
Multi-agent poem analysis module.

Three specialised open-source LLM agents (running via HF Inference API)
replace the previous single-LLM extraction path:

  EmotionAgent   — dominant emotion + nuanced description + intensity
  SemanticAgent  — literary theme + symbolic imagery + cultural context
  MetaphorAgent  — concrete paintable visuals resolved from poetic language

All three run concurrently (asyncio.gather) per poem segment. Any individual
agent failure falls back to the local heuristic for that slot, so a single
agent timeout or rate-limit never blocks the full pipeline.

Entry point for the rest of the codebase: orchestrator.run_agents()
"""
from app.services.agents.orchestrator import AgentResult, run_agents

__all__ = ["AgentResult", "run_agents"]
