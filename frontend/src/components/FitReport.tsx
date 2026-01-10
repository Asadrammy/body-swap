import type { FitReport as FitReportType } from '@/types/api';

interface FitReportProps {
  report: FitReportType;
}

export function FitReport({ report }: FitReportProps) {
  const scaleMap = report.scale_map || {};
  const items = report.items || {};

  const scaleChips = Object.entries(scaleMap)
    .slice(0, 6)
    .map(([key, value]) => {
      const percent = typeof value === 'number' ? `${(value * 100).toFixed(0)}%` : value;
      return { key: key.replace('_', ' '), percent };
    });

  const itemCards = Object.entries(items).map(([item, details]) => {
    const status = details.status || 'scaled';
    const scaleX =
      typeof details.scale_x === 'number' ? `${(details.scale_x * 100).toFixed(0)}%` : '—';
    const scaleY =
      typeof details.scale_y === 'number' ? `${(details.scale_y * 100).toFixed(0)}%` : '—';
    return {
      item: item.replace('_', ' '),
      status,
      scaleX,
      scaleY,
    };
  });

  return (
    <div className="fit-info rounded-3xl bg-brandSurfaceLight dark:bg-brandSurface border border-white/40 dark:border-white/5 p-6 space-y-4">
      <h3 className="text-2xl font-semibold">Clothing Adaptations</h3>
      <div className="space-y-3">
        {scaleChips.length > 0 && (
          <div className="flex flex-wrap gap-2">
            {scaleChips.map(({ key, percent }) => (
              <span
                key={key}
                className="inline-flex items-center rounded-full bg-white/60 dark:bg-white/10 border border-white/80 dark:border-white/10 px-3 py-1 text-xs font-semibold text-slate-700 dark:text-slate-200"
              >
                {key} · {percent}
              </span>
            ))}
          </div>
        )}
        {itemCards.length > 0 ? (
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            {itemCards.map(({ item, status, scaleX, scaleY }) => (
              <div
                key={item}
                className="rounded-2xl bg-white/60 dark:bg-white/5 border border-white/60 dark:border-white/10 p-4"
              >
                <div className="text-xs uppercase tracking-[0.3em] text-slate-500 dark:text-slate-400">
                  {item}
                </div>
                <div className="text-lg font-semibold text-slate-900 dark:text-white">
                  {status === 'scaled' ? `${scaleX} width · ${scaleY} height` : 'No change'}
                </div>
              </div>
            ))}
          </div>
        ) : (
          <p className="text-sm text-slate-500 dark:text-slate-400">
            No clothing adaptations were necessary for this job.
          </p>
        )}
        {report.skin_synthesis_applied && (
          <div className="text-xs text-amber-600 dark:text-amber-300 font-semibold">
            Open-chest region stabilized with skin synthesis.
          </div>
        )}
      </div>
    </div>
  );
}

