import type { BodySummary as BodySummaryType } from '@/types/api';

interface BodySummaryProps {
  summary: BodySummaryType;
}

const measurementLabels: Record<string, string> = {
  shoulder_width: 'Shoulders',
  hip_width: 'Hips',
  waist_width: 'Waist',
  torso_height: 'Torso Height',
  leg_length: 'Leg Length',
  shoulder_hip_ratio: 'Shoulder/Hip Ratio',
};

export function BodySummary({ summary }: BodySummaryProps) {
  const bodyType = summary.body_type ? summary.body_type.replace('_', ' ') : 'Unknown';
  const confidence =
    typeof summary.confidence === 'number'
      ? `${Math.round(Math.max(0, Math.min(1, summary.confidence)) * 100)}%`
      : '—';

  const measurements = summary.measurements || {};
  const measurementEntries = Object.entries(measurementLabels)
    .filter(([key]) => typeof measurements[key] === 'number')
    .slice(0, 4)
    .map(([key, label]) => {
      const value = measurements[key] as number;
      const formatted = key.includes('ratio') ? value.toFixed(2) : `${Math.round(value)}`;
      return { key, label, formatted };
    });

  return (
    <div className="fit-info rounded-3xl bg-brandSurfaceLight dark:bg-brandSurface border border-white/40 dark:border-white/5 p-6 space-y-4">
      <div className="flex items-center gap-3">
        <span className="text-2xl">🧵</span>
        <h3 className="text-2xl font-semibold">Body & Fit Insights</h3>
      </div>
      <div className="space-y-3 text-sm text-slate-700 dark:text-slate-200">
        <div className="flex flex-wrap items-center gap-3">
          <span className="inline-flex items-center rounded-full bg-brandAccent/10 text-brandAccent dark:text-white dark:bg-white/10 px-4 py-1 text-sm font-semibold capitalize">
            {bodyType}
          </span>
          <span className="text-sm text-slate-500 dark:text-slate-300">
            Confidence: <strong>{confidence}</strong>
          </span>
        </div>
        {measurementEntries.length > 0 ? (
          <div className="grid grid-cols-2 gap-3">
            {measurementEntries.map(({ key, label, formatted }) => (
              <div
                key={key}
                className="rounded-2xl bg-white/60 dark:bg-white/5 border border-white/60 dark:border-white/10 p-3"
              >
                <div className="text-xs uppercase tracking-[0.25em] text-slate-500 dark:text-slate-400">
                  {label}
                </div>
                <div className="text-lg font-semibold text-slate-900 dark:text-white">
                  {formatted}
                </div>
              </div>
            ))}
          </div>
        ) : (
          <p className="text-sm text-slate-500 dark:text-slate-400">
            Insufficient measurements.
          </p>
        )}
      </div>
    </div>
  );
}

