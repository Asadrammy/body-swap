# React + TypeScript Frontend Setup Guide

This document explains how to set up and use the new React + TypeScript frontend.

## Overview

The project now includes a modern React frontend built with:
- React 18 + TypeScript
- Vite (build tool)
- React Query (API state management)
- Tailwind CSS (styling)
- Recharts (data visualization)

## Quick Start

### 1. Install Frontend Dependencies

```bash
cd frontend
npm install
```

### 2. Development Mode

**Option A: Separate Dev Servers (Recommended for development)**

Terminal 1 - Start backend:
```bash
python app.py
```

Terminal 2 - Start frontend dev server:
```bash
cd frontend
npm run dev
```

Access frontend at: `http://localhost:3000`

**Option B: Production Mode (Backend serves frontend)**

Build frontend first:
```bash
cd frontend
npm run build
```

Then start backend:
```bash
python app.py
```

Access at: `http://localhost:8000`

### 3. Production Build

```bash
cd frontend
npm run build
```

The built files will be in `static/dist/` and automatically served by FastAPI.

## Migration from Old Frontend

The old HTML/JS frontend (`templates/index.html` and `static/script.js`) is still available as a fallback, but the new React frontend is the recommended approach.

### Key Differences

| Feature | Old Frontend | New React Frontend |
|---------|-------------|-------------------|
| Framework | Vanilla JS | React + TypeScript |
| State Management | Manual | React Query |
| Build Tool | None | Vite |
| Type Safety | None | Full TypeScript |
| Code Organization | Single file | Component-based |
| Development | Edit HTML/JS | Hot reload with Vite |

## Project Structure

```
face-body-swap/
├── frontend/              # New React frontend
│   ├── src/
│   │   ├── components/    # React components
│   │   ├── hooks/         # Custom hooks
│   │   ├── lib/           # API client, utilities
│   │   ├── types/         # TypeScript types
│   │   └── styles/        # CSS
│   ├── package.json
│   └── vite.config.ts
├── static/
│   └── dist/              # Built React app (after npm run build)
├── templates/
│   └── index.html         # Old frontend (fallback)
└── app.py                 # FastAPI backend (serves React app)
```

## Features

### ✅ All Original Features
- File upload (drag & drop)
- Template selection with filtering
- Real-time progress tracking
- Quality metrics visualization
- Body & fit insights
- Dark mode
- Responsive design

### ✅ New Improvements
- Type-safe API integration
- Better error handling
- Component reusability
- Hot module reloading (dev mode)
- Optimized production builds
- Better code organization

## Development Workflow

### Making Changes

1. **Edit React components** in `frontend/src/components/`
2. **Changes auto-reload** in dev mode (`npm run dev`)
3. **Test** at `http://localhost:3000`
4. **Build** when ready: `npm run build`

### Adding New Features

1. Create component in `frontend/src/components/`
2. Add TypeScript types in `frontend/src/types/`
3. Add API hooks in `frontend/src/hooks/` if needed
4. Update `App.tsx` to use new component

## API Integration

The frontend uses React Query for all API calls:

```typescript
// Fetch templates
const { data, isLoading } = useTemplates(category);

// Poll job status
const { data: job } = useJobStatus(jobId, enabled);

// Create swap job
const mutation = useMutation({
  mutationFn: () => swapApi.create(photos, templateId),
  onSuccess: (data) => setJobId(data.job_id),
});
```

All API types match FastAPI schemas for type safety.

## Troubleshooting

### Frontend not building
```bash
cd frontend
rm -rf node_modules package-lock.json
npm install
npm run build
```

### Backend not serving frontend
- Ensure `static/dist/index.html` exists
- Check backend logs for path information
- Verify `app.py` is using the updated code

### Type errors
- Run `npm run build` to see TypeScript errors
- Check that API types match FastAPI schemas
- Ensure all dependencies are installed

### CORS issues
- CORS is already configured in `app.py`
- If issues persist, check browser console
- Verify API URL in `.env` or `vite.config.ts`

## Environment Configuration

Create `frontend/.env`:

```env
VITE_API_URL=http://localhost:8000/api/v1
```

Or use relative URLs (default): `/api/v1`

## Performance

- **Development**: Fast HMR with Vite
- **Production**: Optimized bundle with code splitting
- **API**: Automatic caching and polling with React Query
- **Images**: Optimized loading and preview

## Next Steps

1. ✅ Frontend is ready to use
2. Customize components as needed
3. Add new features using React patterns
4. Deploy with `npm run build` + backend

## Support

For issues:
1. Check browser console (F12)
2. Check backend logs
3. Verify all dependencies installed
4. Ensure backend is running

---

**The React frontend is production-ready and fully integrated with the FastAPI backend!** 🎉

