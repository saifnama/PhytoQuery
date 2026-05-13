import { useEffect, lazy, Suspense } from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Header from './layout/Header';
import { ErrorBoundary } from './ui/ErrorBoundary';
import NerPage from './features/search/NerPage';
import MyPapersPage from './features/papers/MyPapersPage';
import RagPage from './features/chat/RagPage';
import PaperPage from './features/reader/PaperPage';
import { ragApi } from './lib/api';
import { UploadStatusListener } from './components/UploadStatusListener';

// Lazy-load heavy 3D graph — only loaded when user navigates to /dashboard/3d
const Dashboard3D = lazy(() => import('./features/search/Dashboard3D').then(m => ({ default: m.default })));

function App() {
  useEffect(() => {
    const handleBeforeUnload = () => {
      // Fire-and-forget cleanup request using keepalive so it survives tab close
      try {
        ragApi.cleanupUserData();
      } catch {
        // Best-effort cleanup on browser close
      }
    };
    window.addEventListener('beforeunload', handleBeforeUnload);
    return () => window.removeEventListener('beforeunload', handleBeforeUnload);
  }, []);

  return (
    <BrowserRouter>
      {/* Invisible — watches the shared upload store and polls the
          active backend indexing job. Mounted at the layout level so
          uploads triggered from any surface (RagPage, Sidebar) stay
          observed regardless of which route the user is on. */}
      <UploadStatusListener />
      <div className="bg-gray-50 flex h-screen overflow-hidden">
        <main className="flex-1 flex flex-col min-w-0 relative z-10">
          <Header />

          <div className="flex-1 overflow-y-auto relative h-full" id="main-content-display">
            <ErrorBoundary>
              <Routes>
                <Route path="/" element={<NerPage />} />
                <Route path="/mypapers" element={<MyPapersPage />} />
                <Route path="/chat" element={<RagPage />} />
                <Route path="/paper/:doi" element={<PaperPage />} />
                <Route path="/dashboard/3d"
                  element={
                    <Suspense fallback={<div className="flex items-center justify-center h-full"><div className="w-8 h-8 border-4 border-blue-200 border-t-blue-600 rounded-full animate-spin" /></div>}>
                      <Dashboard3D />
                    </Suspense>
                  }
                />
              </Routes>
            </ErrorBoundary>
          </div>
        </main>
      </div>
    </BrowserRouter>
  );
}

export default App;
