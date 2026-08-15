export interface Chapter { start_s: number; end_s: number; title: string; synopsis: string }
export interface KeyQuote { timestamp_s: number; speaker: string | null; text: string }
export interface VisualArtifact {
  timestamp_s: number; kind: 'slide' | 'code' | 'chart' | 'other'
  text: string; description: string | null; frame_path: string | null
}
export interface Report {
  summary: string; chapters: Chapter[]; key_quotes: KeyQuote[]
  action_items: string[]; language: string; trace_id: string; degraded_stages: string[]
  visual_highlights: VisualArtifact[]
}
export interface Job {
  job_id: string; status: 'queued' | 'running' | 'completed' | 'failed'
  report: Report | null; error: string | null
}
export interface TraceSpan {
  stage: string; model_used: string; tokens_in: number; tokens_out: number
  cost_usd: number; latency_ms: number; status: string; fallback_from: string | null
}
export interface Trace { spans: TraceSpan[]; total_cost_usd: number }
export interface StageEvent { stage: string; type: string; message: string | null }
export interface JobOptions { language: string; quality: 'cheap' | 'balanced' | 'best'; force_whisper: boolean; analyze_visuals: boolean }

async function json<T>(respPromise: Promise<Response>): Promise<T> {
  const resp = await respPromise
  if (!resp.ok) throw new Error(`${resp.status} ${await resp.text()}`)
  return resp.json()
}

export const createJob = (url: string, options: JobOptions) =>
  json<{ job_id: string }>(fetch('/api/jobs', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ url, options }),
  }))

export const uploadJob = (file: File, options: JobOptions) => {
  const form = new FormData()
  form.append('file', file)
  form.append('language', options.language)
  form.append('quality', options.quality)
  form.append('force_whisper', String(options.force_whisper))
  form.append('analyze_visuals', String(options.analyze_visuals))
  return json<{ job_id: string }>(fetch('/api/jobs/upload', { method: 'POST', body: form }))
}

export const getJob = (id: string) => json<Job>(fetch(`/api/jobs/${id}`))
export const getTrace = (id: string) => json<Trace>(fetch(`/api/jobs/${id}/trace`))
export const eventsUrl = (id: string) => `/api/jobs/${id}/events`
