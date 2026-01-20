import { useState } from 'react';
import { useMutation } from '@tanstack/react-query';
import { swapApi } from '@/lib/api';
import { Step1Upload } from './components/Step1Upload';
import { Step2Template } from './components/Step2Template';
import { Step3Processing } from './components/Step3Processing';
import { Step4Results } from './components/Step4Results';
import { ThemeToggle } from './components/ThemeToggle';
import type { TemplateMetadata } from '@/types/api';

type Step = 1 | 2 | 3 | 4;

function App() {
  const [currentStep, setCurrentStep] = useState<Step>(1);
  const [photos, setPhotos] = useState<File[]>([]);
  const [selectedTemplate, setSelectedTemplate] = useState<TemplateMetadata | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const [customPrompt, setCustomPrompt] = useState<string>('');

  const swapMutation = useMutation({
    mutationFn: async () => {
      if (!selectedTemplate) {
        throw new Error('No template selected');
      }
      console.log('🚀 [FRONTEND] Starting swap job creation:', {
        photos: photos.map(p => ({ name: p.name, size: p.size, type: p.type })),
        templateId: selectedTemplate.id,
        customPrompt: customPrompt || 'None',
      });
      return swapApi.create(photos, selectedTemplate.id, undefined, customPrompt || undefined);
    },
    onSuccess: (data) => {
      console.log('✅ [FRONTEND] Swap job created successfully:', {
        jobId: data.job_id,
        status: data.status,
        message: data.message,
      });
      setJobId(data.job_id);
      setCurrentStep(3);
    },
    onError: (error) => {
      console.error('❌ [FRONTEND] Swap creation failed:', error);
      alert('Failed to start processing. Please try again.');
      setCurrentStep(1);
    },
  });

  const handleStep1Next = () => {
    console.log('🖱️ [FRONTEND] Step 1 Next clicked:', {
      photosCount: photos.length,
      photos: photos.map(p => ({ name: p.name, size: p.size })),
    });
    if (photos.length > 0) {
      setCurrentStep(2);
    }
  };

  const handleStep2Next = () => {
    console.log('🖱️ [FRONTEND] Step 2 Next clicked (Submit button):', {
      templateId: selectedTemplate?.id,
      templateName: selectedTemplate?.name,
      customPrompt: customPrompt || 'None',
    });
    if (selectedTemplate) {
      swapMutation.mutate();
    }
  };

  const handleProcessingComplete = () => {
    setCurrentStep(4);
  };

  const handleProcessingError = () => {
    // Get the last job error if available
    // Note: We could pass error details through props if needed
    const errorMsg = 'Processing failed. Please check the console for details or try again.';
    alert(errorMsg);
    console.error('Processing failed. Check backend logs for details.');
    setCurrentStep(1);
    resetState();
  };

  const handleCreateNew = () => {
    resetState();
    setCurrentStep(1);
  };

  const resetState = () => {
    setPhotos([]);
    setSelectedTemplate(null);
    setJobId(null);
    setCustomPrompt('');
  };

  return (
    <div className="min-h-screen px-4 py-8">
      <div className="max-w-6xl mx-auto rounded-[40px] bg-white/95 dark:bg-[rgba(8,9,20,0.85)] shadow-[0_40px_120px_rgba(15,23,42,0.35)] border border-white/60 dark:border-white/5 backdrop-blur-3xl ring-1 ring-white/30 dark:ring-white/5">
        {/* Header */}
        <header className="rounded-[30px] bg-gradient-to-br from-brandAccent to-brandAccentAlt text-white p-10 shadow-[0_35px_80px_rgba(102,126,234,0.4)]">
          <div className="space-y-3 max-w-3xl">
            <p className="uppercase tracking-[0.35em] text-white/70 text-sm">Premium AI Studio</p>
            <h1 className="text-4xl sm:text-5xl font-semibold flex items-center gap-3">
              📸 Photo Swap Studio
            </h1>
            <p className="text-white/80 text-lg">
              Transform yourself into any template in seconds with couture-level detailing.
            </p>
          </div>
        </header>

        {/* Main Content */}
        <main className="p-8 space-y-10">
          {currentStep === 1 && (
            <Step1Upload photos={photos} onPhotosChange={setPhotos} onNext={handleStep1Next} />
          )}

          {currentStep === 2 && (
            <Step2Template
              selectedTemplate={selectedTemplate}
              onTemplateSelect={setSelectedTemplate}
              onBack={() => setCurrentStep(1)}
              onNext={handleStep2Next}
              customPrompt={customPrompt}
              onCustomPromptChange={setCustomPrompt}
            />
          )}

          {currentStep === 3 && jobId && (
            <Step3Processing
              jobId={jobId}
              onComplete={handleProcessingComplete}
              onError={handleProcessingError}
            />
          )}

          {currentStep === 4 && jobId && (
            <Step4Results jobId={jobId} onCreateNew={handleCreateNew} />
          )}
        </main>

        {/* Footer */}
        <footer className="text-center text-slate-500 dark:text-slate-400 border-t border-slate-200/70 dark:border-white/10 px-6 py-8">
          <p>&copy; 2025 Photo Swap Studio. All rights reserved.</p>
          <p>Processing powered by AI | Delivery within 24 hours</p>
        </footer>
      </div>

      <ThemeToggle />
    </div>
  );
}

export default App;

