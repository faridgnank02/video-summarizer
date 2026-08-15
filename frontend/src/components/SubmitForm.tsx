import { useState } from 'react'
import { createJob, uploadJob, type JobOptions } from '../api'

export default function SubmitForm({ onSubmitted }: { onSubmitted: (jobId: string) => void }) {
  const [url, setUrl] = useState('')
  const [file, setFile] = useState<File | null>(null)
  const [quality, setQuality] = useState<JobOptions['quality']>('balanced')
  const [language, setLanguage] = useState('en')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  async function submit(e: React.FormEvent) {
    e.preventDefault()
    setBusy(true); setError(null)
    const options: JobOptions = { language, quality, force_whisper: false }
    try {
      const { job_id } = file ? await uploadJob(file, options) : await createJob(url, options)
      onSubmitted(job_id)
    } catch (err) {
      setError(String(err))
    } finally {
      setBusy(false)
    }
  }

  return (
    <form onSubmit={submit} className="space-y-4 rounded-xl border border-slate-200 p-6">
      <input
        className="w-full rounded-lg border border-slate-300 px-3 py-2"
        placeholder="YouTube URL"
        value={url}
        onChange={(e) => setUrl(e.target.value)}
      />
      <div className="flex items-center gap-4 text-sm">
        <span className="text-slate-500">or upload:</span>
        <input type="file" accept="video/*,audio/*" onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
      </div>
      <div className="flex gap-4">
        <select className="rounded-lg border border-slate-300 px-2 py-1"
                value={quality} onChange={(e) => setQuality(e.target.value as JobOptions['quality'])}>
          <option value="cheap">Cheap</option>
          <option value="balanced">Balanced</option>
          <option value="best">Best</option>
        </select>
        <select className="rounded-lg border border-slate-300 px-2 py-1"
                value={language} onChange={(e) => setLanguage(e.target.value)}>
          <option value="en">English</option>
          <option value="fr">Français</option>
        </select>
        <button disabled={busy || (!url && !file)}
                className="rounded-lg bg-slate-900 px-4 py-1.5 text-white disabled:opacity-40">
          {busy ? 'Submitting…' : 'Analyze'}
        </button>
      </div>
      {error && <p className="text-sm text-red-600">{error}</p>}
    </form>
  )
}
