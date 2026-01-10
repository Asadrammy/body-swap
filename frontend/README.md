# Face & Body Swap - React Frontend

Modern React + TypeScript frontend for the Face & Body Swap application.

## Tech Stack

- **React 18+** - UI framework
- **TypeScript** - Type safety
- **Vite** - Build tool and dev server
- **React Query** - API state management and caching
- **Tailwind CSS** - Styling
- **React Hook Form** - Form handling
- **Recharts** - Quality metrics visualization
- **React Dropzone** - File uploads

## Getting Started

### Prerequisites

- Node.js 18+ and npm/yarn/pnpm
- Python backend running on port 8000 (or configure `VITE_API_URL`)

### Installation

```bash
cd frontend
npm install
```

### Development

Start the development server (with hot reload):

```bash
npm run dev
```

The frontend will be available at `http://localhost:3000` and will proxy API requests to `http://localhost:8000`.

### Building for Production

Build the frontend for production:

```bash
npm run build
```

This will create optimized production files in `../static/dist/` which the FastAPI backend will serve.

### Project Structure

```
frontend/
├── src/
│   ├── components/       # React components
│   │   ├── Step1Upload.tsx
│   │   ├── Step2Template.tsx
│   │   ├── Step3Processing.tsx
│   │   ├── Step4Results.tsx
│   │   ├── QualityMetrics.tsx
│   │   ├── BodySummary.tsx
│   │   ├── FitReport.tsx
│   │   └── ThemeToggle.tsx
│   ├── hooks/           # Custom React hooks
│   │   ├── useTemplates.ts
│   │   ├── useJobStatus.ts
│   │   └── useTheme.ts
│   ├── lib/             # Utilities and API client
│   │   ├── api.ts
│   │   └── react-query.ts
│   ├── types/           # TypeScript type definitions
│   │   └── api.ts
│   ├── styles/          # Global styles
│   │   └── index.css
│   ├── App.tsx          # Main app component
│   └── main.tsx         # Entry point
├── package.json
├── tsconfig.json
├── vite.config.ts
└── tailwind.config.js
```

## Features

### Step 1: Upload Photos
- Drag & drop file upload
- Support for 1-2 customer photos
- Image preview with remove option
- File validation (max 10MB, images only)

### Step 2: Template Selection
- Template gallery with categories
- Filter by: All, Individual, Couples, Family
- Template preview cards
- Selection highlighting

### Step 3: Processing
- Real-time progress tracking
- Status updates via polling
- Stage information
- Estimated time display

### Step 4: Results
- Result image display
- Quality metrics visualization (Recharts)
- Body & fit insights
- Download options (image, bundle)
- Share functionality
- Create another option

### Additional Features
- Dark mode support with system preference detection
- Responsive design (mobile-friendly)
- Type-safe API integration
- Error handling and loading states

## API Integration

The frontend uses React Query for API state management:

- **Templates**: `useTemplates()` hook for fetching templates
- **Job Status**: `useJobStatus()` hook with automatic polling
- **Swap Creation**: Mutation hook for creating swap jobs

All API calls are type-safe using TypeScript interfaces that match the FastAPI schemas.

## Environment Variables

Create a `.env` file in the `frontend/` directory:

```env
VITE_API_URL=http://localhost:8000/api/v1
```

If not set, defaults to `/api/v1` (relative URL for same-origin requests).

## Development Workflow

1. **Start backend** (in project root):
   ```bash
   python app.py
   ```

2. **Start frontend dev server** (in frontend/):
   ```bash
   npm run dev
   ```

3. **Access frontend**: `http://localhost:3000`

The Vite dev server will proxy API requests to the backend automatically.

## Production Deployment

1. **Build frontend**:
   ```bash
   cd frontend
   npm run build
   ```

2. **Start backend** (serves built frontend):
   ```bash
   python app.py
   ```

3. **Access application**: `http://localhost:8000`

The FastAPI backend will serve the built React app from `static/dist/`.

## Troubleshooting

### Frontend not loading
- Ensure you've run `npm run build` in the frontend directory
- Check that `static/dist/index.html` exists
- Verify backend is running and serving static files correctly

### API requests failing
- Check backend is running on port 8000
- Verify CORS is enabled in FastAPI (already configured)
- Check browser console for errors

### Type errors
- Run `npm run build` to check for TypeScript errors
- Ensure all dependencies are installed: `npm install`

## Scripts

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run preview` - Preview production build
- `npm run lint` - Run ESLint

## License

Same as main project.

