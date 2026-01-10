import { useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { Upload, X } from 'lucide-react';

interface Step1UploadProps {
  photos: File[];
  onPhotosChange: (photos: File[]) => void;
  onNext: () => void;
}

export function Step1Upload({ photos, onPhotosChange, onNext }: Step1UploadProps) {
  const onDrop = useCallback(
    (acceptedFiles: File[]) => {
      const validFiles = acceptedFiles.filter(
        (file) => file.type.startsWith('image/') && file.size <= 10 * 1024 * 1024
      );

      if (validFiles.length === 0) {
        alert('Please select valid image files (max 10MB each)');
        return;
      }

      if (validFiles.length > 2) {
        alert('Maximum 2 photos allowed');
        return;
      }

      onPhotosChange(validFiles);
    },
    [onPhotosChange]
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.jpg', '.jpeg', '.png'],
    },
    maxFiles: 2,
  });

  const removePhoto = (index: number) => {
    const newPhotos = photos.filter((_, i) => i !== index);
    onPhotosChange(newPhotos);
  };

  const clearAll = () => {
    onPhotosChange([]);
  };

  return (
    <div className="card space-y-6">
      <div>
        <h2 className="text-3xl font-semibold">Step 1: Upload Your Photos</h2>
        <p className="text-slate-600 dark:text-slate-300 mt-2">
          Upload 1-2 photos of yourself (or couple/family)
        </p>
      </div>

      <div
        {...getRootProps()}
        className={`group relative flex flex-col items-center justify-center rounded-3xl border-2 border-dashed ${
          isDragActive
            ? 'border-brandAccentAlt bg-brandAccent/20'
            : 'border-brandAccent bg-brandAccent/5 dark:bg-brandSurface/30'
        } px-8 py-16 text-center transition hover:border-brandAccentAlt hover:bg-brandAccent/10 cursor-pointer`}
      >
        <input {...getInputProps()} />
        <div className="text-6xl mb-6">📷</div>
        <p className="text-lg font-medium mb-2">
          {isDragActive
            ? 'Drop your photos here'
            : 'Drag & drop your photos here or click to browse'}
        </p>
        <p className="text-sm text-slate-500 dark:text-slate-300 mb-6">
          Accepted: JPG, PNG (Max 10MB each)
        </p>
        <button
          type="button"
          className="btn-primary inline-flex items-center gap-2"
          onClick={(e) => {
            e.stopPropagation();
            // Trigger file input
            const input = document.querySelector('input[type="file"]') as HTMLInputElement;
            input?.click();
          }}
        >
          <Upload className="w-5 h-5" />
          Choose Photos
        </button>
      </div>

      {photos.length > 0 && (
        <div className="flex flex-wrap gap-6">
          {photos.map((photo, index) => (
            <div key={index} className="relative group">
              <img
                src={URL.createObjectURL(photo)}
                alt={`Photo ${index + 1}`}
                className="w-32 h-32 object-cover rounded-2xl shadow-lg"
              />
              <button
                onClick={() => removePhoto(index)}
                className="absolute -top-2 -right-2 w-8 h-8 rounded-full bg-red-500 text-white flex items-center justify-center hover:bg-red-600 transition-colors shadow-lg"
                aria-label="Remove photo"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
          ))}
        </div>
      )}

      <div className="flex flex-wrap gap-4">
        <button type="button" onClick={clearAll} className="btn-secondary">
          Clear
        </button>
        <button
          type="button"
          onClick={onNext}
          disabled={photos.length === 0}
          className="btn-primary"
        >
          Next: Select Template →
        </button>
      </div>
    </div>
  );
}

