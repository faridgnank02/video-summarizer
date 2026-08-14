import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import ReportView from './ReportView'
import type { Report } from '../api'

const report: Report = {
  summary: 'A great talk about pipelines.',
  chapters: [{ start_s: 0, end_s: 90, title: 'Introduction', synopsis: 'Opening.' }],
  key_quotes: [{ timestamp_s: 42, speaker: null, text: 'Ship it.' }],
  action_items: ['Try the demo'],
  language: 'en',
  trace_id: 'tr1',
  degraded_stages: [],
  visual_highlights: [
    { timestamp_s: 30, kind: 'slide', text: 'Roadmap 2026', description: null, frame_path: null },
    { timestamp_s: 90, kind: 'chart', text: '', description: 'Revenue bar chart', frame_path: null },
  ],
}

describe('ReportView', () => {
  it('renders summary, chapters with timestamps, quotes, and action items', () => {
    render(<ReportView report={report} />)
    expect(screen.getByText('A great talk about pipelines.')).toBeInTheDocument()
    expect(screen.getByText('Introduction')).toBeInTheDocument()
    expect(screen.getByText('0:00')).toBeInTheDocument()
    expect(screen.getByText('"Ship it."')).toBeInTheDocument()
    expect(screen.getByText('Try the demo')).toBeInTheDocument()
  })

  it('shows a degraded banner when stages were skipped', () => {
    render(<ReportView report={{ ...report, degraded_stages: ['chapterize'] }} />)
    expect(screen.getByText(/degraded/i)).toBeInTheDocument()
  })

  it('renders the visuals section with kinds and text', () => {
    render(<ReportView report={report} />)
    expect(screen.getByText('Visuals')).toBeInTheDocument()
    expect(screen.getByText('Roadmap 2026')).toBeInTheDocument()
    expect(screen.getByText('Revenue bar chart')).toBeInTheDocument()
  })
})
