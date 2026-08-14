import { useEffect, useState } from 'react'
import { eventsUrl, type StageEvent } from '../api'

const STAGES = ['ingest', 'transcribe', 'chapterize', 'synthesize']

export default function JobProgress({ jobId, onFinished }: { jobId: string; onFinished: () => void }) {
  const [events, setEvents] = useState<StageEvent[]>([])

  useEffect(() => {
    const es = new EventSource(eventsUrl(jobId))
    es.onmessage = (msg) => {
      const ev: StageEvent = JSON.parse(msg.data)
      setEvents((prev) => [...prev, ev])
      if (ev.stage === 'pipeline' || ev.type === 'failed') {
        es.close()
        onFinished()
      }
    }
    es.onerror = () => { es.close(); onFinished() }
    return () => es.close()
  }, [jobId, onFinished])

  const statusOf = (stage: string) => {
    const evs = events.filter((e) => e.stage === stage)
    if (evs.some((e) => e.type === 'completed')) return '✓'
    if (evs.some((e) => e.type === 'failed')) return '✗'
    if (evs.some((e) => e.type === 'started')) return '…'
    return '·'
  }

  return (
    <ol className="flex gap-6 text-sm">
      {STAGES.map((s) => (
        <li key={s} className="flex items-center gap-2">
          <span className="font-mono">{statusOf(s)}</span>
          <span className="capitalize">{s}</span>
        </li>
      ))}
    </ol>
  )
}
