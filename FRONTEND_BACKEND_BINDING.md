# Frontend-Backend Binding Documentation

## Overview

This document explains how the frontend React components bind with the backend FastAPI routes for the face-body swap pipeline.

## Data Flow

### Step 1: Upload Customer Photos (Frontend)

**Component**: `Step1Upload.tsx`

**Frontend Action**:
- User uploads 1-2 customer photos via drag-and-drop or file picker
- Files stored in React state: `photos: File[]`

**Data Structure**:
```typescript
photos: File[]  // Array of File objects (1-2 images)
```

---

### Step 2: Select Template (Frontend)

**Component**: `Step2Template.tsx`

**Frontend Actions**:
1. User selects template from gallery (`selectedTemplate: TemplateMetadata`)
2. Optional: User provides custom AI prompt (`customPrompt: string`)

**API Call**: `GET /api/v1/templates`
- Fetches available templates
- Filtered by category (all, individual, couple, family)

**Data Structure**:
```typescript
selectedTemplate: {
  id: string
  name: string
  category: string
  preview_url: string
  asset_path: string
  // ... other metadata
}

customPrompt: string  // Optional custom prompt text
```

---

### Step 3: Submit Swap Request (Frontend → Backend)

**Component**: `App.tsx` → `swapApi.create()`

**Frontend API Call** (`frontend/src/lib/api.ts`):
```typescript
swapApi.create(
  customerPhotos: File[],
  templateId?: string,
  templateFile?: File,
  customPrompt?: string
)
```

**Request Format**:
- **Method**: `POST`
- **Endpoint**: `/api/v1/swap`
- **Content-Type**: `multipart/form-data`
- **Body**:
  - `customer_photos`: File[] (1-2 files)
  - `template_id`: string (optional, if using template from catalog)
  - `template`: File (optional, if uploading custom template)
  - `custom_prompt`: string (optional)

**Backend Route** (`src/api/routes.py`):
```python
@router.post("/swap", response_model=SwapResponse)
async def create_swap_job(
    background_tasks: BackgroundTasks,
    customer_photos: List[UploadFile] = File(...),
    template: Optional[UploadFile] = File(None),
    template_id: Optional[str] = Form(None),
    custom_prompt: Optional[str] = Form(None)
)
```

**Backend Processing**:
1. Validates input (1-2 photos, template required)
2. Generates unique `job_id` (UUID)
3. Saves uploaded files to `temp/` directory:
   - `{job_id}_customer_0.jpg`, `{job_id}_customer_1.jpg`
   - `{job_id}_template.jpg`
4. If `template_id` provided, resolves path from template catalog
5. Creates job entry in `jobs` dict:
   ```python
   jobs[job_id] = {
       "job_id": job_id,
       "status": "pending",
       "progress": 0.0,
       "custom_prompt": custom_prompt,  # Stored here
       "created_at": datetime.now(),
       "updated_at": datetime.now()
   }
   ```
6. Starts background processing task:
   ```python
   pipeline.process(
       job_id,
       customer_paths,      # List[str] - file paths
       str(template_path),  # str - template file path
       custom_prompt=custom_prompt  # Optional[str] - passed to pipeline
   )
   ```

**Response**:
```typescript
{
  job_id: string
  status: "pending"
  message: string
  created_at: string
}
```

---

### Step 4: Processing (Frontend Polling)

**Component**: `Step3Processing.tsx`

**Frontend API Call**: `GET /api/v1/jobs/{jobId}`

**Backend Route**:
```python
@router.get("/jobs/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
```

**Response Structure**:
```typescript
{
  job_id: string
  status: "pending" | "processing" | "completed" | "failed"
  progress: number  // 0-1
  current_stage?: string
  error?: string
  body_summary?: BodySummary
  fit_report?: FitReport
  quality_metrics?: QualityMetrics
  refinement_masks?: Record<string, string>
  created_at: string
  updated_at: string
}
```

**Frontend Polling**:
- Polls every 2 seconds using `useJobStatus` hook
- Updates progress bar and stage text
- Moves to Step 4 when `status === "completed"`

---

### Step 5: Results (Frontend Display)

**Component**: `Step4Results.tsx`

**Frontend API Calls**:
1. **Get Result Image**: `GET /api/v1/jobs/{jobId}/result`
   - Returns: `Blob` (image file)
   - Backend: `FileResponse` with saved result image

2. **Download Bundle**: `GET /api/v1/jobs/{jobId}/bundle`
   - Returns: `Blob` (ZIP file with result + masks)
   - Backend: `StreamingResponse` with ZIP archive

3. **Get Status**: `GET /api/v1/jobs/{jobId}`
   - Returns: `JobStatus` with quality metrics, refinement masks, etc.

---

## Custom Prompt Flow

### Frontend → Backend

1. **User Input** (`Step2Template.tsx`):
   ```tsx
   <textarea
     value={customPrompt}
     onChange={(e) => onCustomPromptChange?.(e.target.value)}
   />
   ```

2. **State Management** (`App.tsx`):
   ```typescript
   const [customPrompt, setCustomPrompt] = useState<string>('');
   ```

3. **API Call** (`App.tsx`):
   ```typescript
   swapApi.create(photos, selectedTemplate.id, undefined, customPrompt || undefined)
   ```

4. **FormData** (`api.ts`):
   ```typescript
   if (customPrompt) {
     formData.append('custom_prompt', customPrompt);
   }
   ```

5. **Backend Receives** (`routes.py`):
   ```python
   custom_prompt: Optional[str] = Form(None)
   ```

6. **Stored in Job**:
   ```python
   jobs[job_id]["custom_prompt"] = custom_prompt
   ```

7. **Passed to Pipeline**:
   ```python
   pipeline.process(..., custom_prompt=custom_prompt)
   ```

### Backend Processing → Refiner

1. **Pipeline Receives** (`routes.py:process()`):
   ```python
   def process(self, job_id: str, ..., custom_prompt: Optional[str] = None):
       # custom_prompt is passed directly
   ```

2. **Passed to Refiner**:
   ```python
   refined = self.refiner.refine_composition(
       composed,
       template_analysis,
       fused_body_shape,
       strength=0.8,
       custom_prompt=custom_prompt  # Passed here
   )
   ```

3. **Refiner Uses Custom Prompt** (`refiner.py`):
   ```python
   def refine_composition(..., custom_prompt: Optional[str] = None):
       if custom_prompt:
           prompt = custom_prompt  # Use custom prompt
           logger.info(f"Using custom prompt: {custom_prompt[:100]}...")
       else:
           prompt = self._generate_refinement_prompt(...)  # Auto-generate
   ```

4. **Used in Stability AI API**:
   ```python
   refined = self.generator.refine(
       image=working,
       prompt=prompt,  # Custom or auto-generated
       mask=refinement_mask,
       negative_prompt=negative_prompt,
       ...
   )
   ```

---

## Key Bindings Summary

| Frontend | Backend | Binding Type |
|----------|---------|--------------|
| `File[]` customer photos | `List[UploadFile]` | `multipart/form-data` |
| `string` template_id | `Optional[str]` Form | `Form(None)` |
| `File` template (optional) | `Optional[UploadFile]` | `File(None)` |
| `string` custom_prompt | `Optional[str]` Form | `Form(None)` |
| `string` job_id | `str` job_id | URL path parameter |
| `JobStatus` response | `JobStatus` model | JSON serialization |
| `Blob` result image | `FileResponse` | Binary file stream |
| `Blob` bundle ZIP | `StreamingResponse` | ZIP archive stream |

---

## Error Handling

### Frontend
- API errors caught in `swapMutation.onError`
- Shows alert and resets to Step 1
- Logs error to console

### Backend
- Validation errors: `HTTPException(status_code=400, ...)`
- File not found: `HTTPException(status_code=404, ...)`
- Processing errors: Job status set to `"failed"`, error stored in `jobs[job_id]["error"]`

---

## Testing the Binding

### Manual Test Flow

1. **Frontend Upload**:
   ```typescript
   // In browser console
   const formData = new FormData();
   formData.append('customer_photos', file1);
   formData.append('customer_photos', file2);  // Optional
   formData.append('template_id', 'template-1');
   formData.append('custom_prompt', 'photorealistic portrait, natural lighting');
   
   await fetch('/api/v1/swap', {
     method: 'POST',
     body: formData
   });
   ```

2. **Check Job Status**:
   ```bash
   curl http://localhost:8000/api/v1/jobs/{job_id}
   ```

3. **Verify Custom Prompt**:
   ```bash
   # Should see in backend logs:
   # "Using custom prompt: photorealistic portrait, natural lighting..."
   ```

---

## Configuration

### Frontend API Base URL

**File**: `frontend/src/lib/api.ts`

```typescript
const API_BASE_URL = import.meta.env.VITE_API_URL || '/api/v1';
```

**Environment Variable**: `VITE_API_URL`
- Default: `/api/v1` (relative, uses same origin)
- Can be set to absolute URL: `http://localhost:8000/api/v1`

### Backend CORS

**File**: `src/api/main.py`

Should include CORS middleware for frontend origin if different:
```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite default
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## Status: ✅ Binding Complete

All bindings are correctly implemented:
- ✅ Frontend FormData → Backend multipart/form-data
- ✅ Custom prompt passed through entire pipeline
- ✅ Job status polling works correctly
- ✅ File upload/download works
- ✅ Error handling in place

No changes needed - the binding is correct!

