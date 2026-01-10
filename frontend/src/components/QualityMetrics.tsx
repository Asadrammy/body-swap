import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';
import type { QualityMetrics as QualityMetricsType } from '@/types/api';

interface QualityMetricsProps {
  metrics: QualityMetricsType;
}

const metricNames: Record<string, string> = {
  overall_score: 'Overall Quality',
  face_similarity: 'Face Similarity',
  pose_accuracy: 'Pose Accuracy',
  clothing_fit: 'Clothing Fit',
  seamless_blending: 'Seamless Blending',
  sharpness: 'Sharpness',
};

export function QualityMetrics({ metrics }: QualityMetricsProps) {
  const chartData = Object.entries(metrics)
    .filter(([key, value]) => typeof value === 'number' && metricNames[key])
    .map(([key, value]) => ({
      name: metricNames[key],
      value: (value as number) * 100,
      fullValue: value as number,
    }));

  return (
    <div className="quality-info rounded-3xl bg-brandSurfaceLight dark:bg-brandSurface border border-white/40 dark:border-white/5 p-6 space-y-4">
      <h3 className="text-2xl font-semibold">Quality Score</h3>
      {chartData.length > 0 ? (
        <div className="space-y-4">
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={chartData}>
              <XAxis dataKey="name" angle={-45} textAnchor="end" height={80} fontSize={12} />
              <YAxis domain={[0, 100]} />
              <Tooltip formatter={(value: number) => `${value.toFixed(0)}%`} />
              <Bar dataKey="value" fill="#667eea" />
            </BarChart>
          </ResponsiveContainer>
          <div className="grid grid-cols-2 gap-3">
            {chartData.map((item) => (
              <div
                key={item.name}
                className="rounded-2xl bg-white/60 dark:bg-white/5 border border-white/60 dark:border-white/10 p-3"
              >
                <div className="text-xs uppercase tracking-[0.25em] text-slate-500 dark:text-slate-400">
                  {item.name}
                </div>
                <div className="text-lg font-semibold text-slate-900 dark:text-white">
                  {item.value.toFixed(0)}%
                </div>
              </div>
            ))}
          </div>
        </div>
      ) : (
        <p className="text-sm text-slate-500 dark:text-slate-400">
          Quality metrics will appear here once available.
        </p>
      )}
    </div>
  );
}

