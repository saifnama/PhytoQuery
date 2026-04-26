import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Header from './layout/Header';
import { ErrorBoundary } from './ui/ErrorBoundary';
import NerPage from './features/search/NerPage';
import MyPapersPage from './features/papers/MyPapersPage';
import RagPage from './features/chat/RagPage';
import PaperPage from './features/reader/PaperPage';

function App() {
  return (
    <BrowserRouter>
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
              </Routes>
            </ErrorBoundary>
          </div>
        </main>
      </div>
    </BrowserRouter>
  );
}

export default App;
