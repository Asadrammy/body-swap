import { useState } from 'react';
import { useTemplates } from '@/hooks/useTemplates';
import type { TemplateMetadata } from '@/types/api';
import { Loader2 } from 'lucide-react';

interface Step2TemplateProps {
  selectedTemplate: TemplateMetadata | null;
  onTemplateSelect: (template: TemplateMetadata) => void;
  onBack: () => void;
  onNext: () => void;
  customPrompt?: string;
  onCustomPromptChange?: (prompt: string) => void;
}

const categories = [
  { id: 'all', label: 'All' },
  { id: 'individual', label: 'Individual' },
  { id: 'couple', label: 'Couples' },
  { id: 'family', label: 'Family' },
];

export function Step2Template({
  selectedTemplate,
  onTemplateSelect,
  onBack,
  onNext,
  customPrompt = '',
  onCustomPromptChange,
}: Step2TemplateProps) {
  const [selectedCategory, setSelectedCategory] = useState<string>('all');
  const { data, isLoading, error } = useTemplates(
    selectedCategory === 'all' ? undefined : selectedCategory
  );

  const templates = data?.templates || [];

  return (
    <div className="card space-y-6">
      <div>
        <h2 className="text-3xl font-semibold">Step 2: Choose Your Template</h2>
        <p className="text-slate-600 dark:text-slate-300 mt-2">
          Select the style, pose, and clothing you want
        </p>
      </div>

      {/* Categories */}
      <div className="flex flex-wrap gap-3">
        {categories.map((cat) => (
          <button
            key={cat.id}
            type="button"
            onClick={() => setSelectedCategory(cat.id)}
            className={`rounded-full border px-5 py-2 transition ${
              selectedCategory === cat.id
                ? 'border-brandAccent bg-brandAccent text-white dark:border-white/30'
                : 'border-brandAccent text-brandAccent hover:bg-brandAccent hover:text-white dark:text-white dark:border-white/30'
            }`}
          >
            {cat.label}
          </button>
        ))}
      </div>

      {/* Template Gallery */}
      <div className="template-gallery">
        {isLoading && (
          <div className="col-span-full text-center text-slate-500 dark:text-slate-300 py-10 flex items-center justify-center gap-2">
            <Loader2 className="w-5 h-5 animate-spin" />
            Loading templates...
          </div>
        )}

        {error && (
          <div className="col-span-full text-center text-red-500 py-10">
            Failed to load templates. Please try again.
          </div>
        )}

        {!isLoading && !error && templates.length === 0 && (
          <div className="col-span-full text-center text-slate-500 dark:text-slate-300 py-10">
            No templates found for this category.
          </div>
        )}

        {!isLoading && templates.length > 0 && (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
            {templates.map((template) => (
              <div
                key={template.id}
                onClick={() => onTemplateSelect(template)}
                className={`template-card rounded-2xl overflow-hidden border-2 cursor-pointer transition-all ${
                  selectedTemplate?.id === template.id
                    ? 'border-brandAccent ring-4 ring-brandAccent/20 shadow-lg'
                    : 'border-white/50 dark:border-white/10 hover:border-brandAccent/50'
                }`}
              >
                <div className="aspect-square overflow-hidden bg-slate-200 dark:bg-slate-800">
                  <img
                    src={template.preview_url}
                    alt={template.name}
                    className="w-full h-full object-cover"
                  />
                </div>
                <div className="p-4 bg-white/90 dark:bg-brandDark/70">
                  <h3 className="font-semibold text-lg mb-2">{template.name}</h3>
                  {template.tags.length > 0 && (
                    <div className="flex flex-wrap gap-2">
                      {template.tags.slice(0, 3).map((tag, idx) => (
                        <span
                          key={idx}
                          className="text-xs px-2 py-1 rounded-full bg-brandAccent/10 text-brandAccent dark:bg-white/10 dark:text-white"
                        >
                          {tag}
                        </span>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Custom Prompt Field */}
      <div className="space-y-3">
        <label className="block text-lg font-semibold">
          Custom AI Prompt (Optional)
        </label>
        <p className="text-sm text-slate-600 dark:text-slate-300">
          Provide a custom prompt to guide AI image manipulation. Leave empty to use auto-generated prompts.
        </p>
        <textarea
          value={customPrompt}
          onChange={(e) => onCustomPromptChange?.(e.target.value)}
          placeholder="e.g., photorealistic portrait, natural lighting, professional photography, high quality, detailed features..."
          className="w-full px-4 py-3 rounded-2xl border border-brandAccent/30 dark:border-white/20 bg-white dark:bg-brandSurface focus:ring-2 focus:ring-brandAccent focus:border-brandAccent outline-none transition"
          rows={3}
        />
      </div>

      <div className="flex flex-wrap gap-4">
        <button type="button" onClick={onBack} className="btn-secondary">
          ← Back
        </button>
        <button
          type="button"
          onClick={onNext}
          disabled={!selectedTemplate}
          className="btn-primary"
        >
          Next: Process →
        </button>
      </div>
    </div>
  );
}

