-- Export one call's agent events from the reference product's Neon database (AdithyahNair/take2VoiceAgent, apps/web)
-- so the post-interview analysis can read it:  psql "$DATABASE_URL_READONLY" -At -v call='<call_id>' -f export_call.sql > call.json
-- Then: uv run python evals/ai_detect/analyze_call.py call.json   (or drop the file into calls/ for the dashboard)
select json_agg(json_build_object('seq', seq, 'type', type, 'payload', payload, 'call_id', call_id) order by seq)
from agent_events
where call_id = :'call';
