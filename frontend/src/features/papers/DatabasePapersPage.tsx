import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { dbApi } from '../../lib/api';
import { Database, Article, CaretLeft, CaretRight } from '@phosphor-icons/react';

interface DBPaper {
  id: number;
  doi: string;
  title: string;
  journal: string | null;
  year: number | null;
  is_open_access: boolean | null;
  entity_count: number;
}

const DatabasePapersPage: React.FC = () => {
  const navigate = useNavigate();
  const [papers, setPapers] = useState<DBPaper[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  
  // Pagination
  const [page, setPage] = useState(1);
  const [totalCount, setTotalCount] = useState(0);
  const limit = 20;

  useEffect(() => {
    const fetchPapers = async () => {
      setIsLoading(true);
      try {
        const offset = (page - 1) * limit;
        const data = await dbApi.getPapers(limit, offset);
        setPapers(data.papers);
        setTotalCount(data.total);
        setError(null);
      } catch (err: any) {
        console.error('Failed to fetch database papers:', err);
        setError('Failed to load papers from the local database. Ensure the backend is running.');
      } finally {
        setIsLoading(false);
      }
    };

    fetchPapers();
  }, [page]);

  const totalPages = Math.ceil(totalCount / limit);

  return (
    <div className="flex flex-col h-full bg-slate-50 overflow-hidden">
      <div className="flex-shrink-0 bg-white border-b border-slate-200 px-8 py-6 flex items-center justify-between">
        <div className="flex items-center gap-4">
          <div className="h-12 w-12 rounded-xl bg-emerald-100 flex items-center justify-center text-emerald-600">
            <Database size={24} weight="duotone" />
          </div>
          <div>
            <h1 className="text-2xl font-bold text-slate-900 font-display">Database Papers</h1>
            <p className="text-sm text-slate-500 mt-1">Browse all {totalCount.toLocaleString()} papers synced in your local SQLite database</p>
          </div>
        </div>
      </div>

      <div className="flex-1 overflow-auto p-8">
        <div className="max-w-6xl mx-auto">
          {error && (
            <div className="bg-red-50 text-red-600 p-4 rounded-xl border border-red-100 mb-6 flex items-start gap-3">
              <div className="mt-0.5"><Database size={18} /></div>
              <p className="text-sm">{error}</p>
            </div>
          )}

          <div className="bg-white rounded-2xl shadow-sm border border-slate-200 overflow-hidden">
            <div className="overflow-x-auto">
              <table className="w-full text-left text-sm text-slate-600">
                <thead className="bg-slate-50 text-slate-900 font-semibold border-b border-slate-200">
                  <tr>
                    <th className="px-6 py-4 w-12">ID</th>
                    <th className="px-6 py-4">Title & Journal</th>
                    <th className="px-6 py-4 w-32">Year</th>
                    <th className="px-6 py-4 w-32 text-right">Entities</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100">
                  {isLoading ? (
                    // Skeleton rows
                    Array.from({ length: 10 }).map((_, i) => (
                      <tr key={i} className="animate-pulse">
                        <td className="px-6 py-4"><div className="h-4 bg-slate-100 rounded w-8"></div></td>
                        <td className="px-6 py-4">
                          <div className="h-4 bg-slate-100 rounded w-3/4 mb-2"></div>
                          <div className="h-3 bg-slate-50 rounded w-1/2"></div>
                        </td>
                        <td className="px-6 py-4"><div className="h-4 bg-slate-100 rounded w-12"></div></td>
                        <td className="px-6 py-4 flex justify-end"><div className="h-5 bg-slate-100 rounded-full w-16"></div></td>
                      </tr>
                    ))
                  ) : papers.length === 0 ? (
                    <tr>
                      <td colSpan={4} className="px-6 py-12 text-center text-slate-500">
                        <Database size={32} className="mx-auto mb-3 text-slate-300" />
                        <p>No papers found in the database.</p>
                      </td>
                    </tr>
                  ) : (
                    papers.map((paper) => (
                      <tr 
                        key={paper.id} 
                        onClick={() => navigate(`/paper/${encodeURIComponent(paper.doi || paper.id.toString())}`)}
                        className="hover:bg-blue-50 cursor-pointer transition-colors group"
                      >
                        <td className="px-6 py-4 align-top font-mono text-xs text-slate-400 pt-5">
                          #{paper.id}
                        </td>
                        <td className="px-6 py-4 align-top">
                          <div className="font-medium text-slate-900 group-hover:text-blue-700 transition-colors mb-1">
                            {paper.title ? (
                              <span dangerouslySetInnerHTML={{ __html: paper.title }} />
                            ) : (
                              <span className="text-slate-400 italic">Untitled Paper</span>
                            )}
                          </div>
                          <div className="flex items-center gap-2 text-xs text-slate-500">
                            {paper.journal && <span className="flex items-center gap-1"><Article size={12} /> {paper.journal}</span>}
                            {paper.doi && <span className="text-slate-400 font-mono">doi:{paper.doi}</span>}
                            {paper.is_open_access && (
                              <span className="bg-emerald-50 text-emerald-600 px-1.5 py-0.5 rounded text-[10px] font-bold tracking-wide uppercase">
                                Open Access
                              </span>
                            )}
                          </div>
                        </td>
                        <td className="px-6 py-4 align-top pt-5">
                          {paper.year ? (
                            <span className="bg-slate-100 text-slate-600 px-2 py-1 rounded text-xs font-mono">
                              {paper.year}
                            </span>
                          ) : (
                            <span className="text-slate-300 text-xs">—</span>
                          )}
                        </td>
                        <td className="px-6 py-4 align-top text-right pt-5">
                          {paper.entity_count > 0 ? (
                            <span className="inline-flex items-center justify-center px-2 py-1 bg-violet-50 text-violet-700 text-xs font-bold rounded-full border border-violet-100">
                              {paper.entity_count}
                            </span>
                          ) : (
                            <span className="text-slate-300 text-xs">—</span>
                          )}
                        </td>
                      </tr>
                    ))
                  )}
                </tbody>
              </table>
            </div>
            
            {/* Pagination Controls */}
            {!isLoading && totalPages > 1 && (
              <div className="bg-slate-50 border-t border-slate-200 px-6 py-4 flex items-center justify-between">
                <span className="text-sm text-slate-500">
                  Showing <span className="font-medium text-slate-900">{(page - 1) * limit + 1}</span> to{' '}
                  <span className="font-medium text-slate-900">{Math.min(page * limit, totalCount)}</span> of{' '}
                  <span className="font-medium text-slate-900">{totalCount.toLocaleString()}</span> papers
                </span>
                <div className="flex items-center gap-2">
                  <button
                    onClick={() => setPage(p => Math.max(1, p - 1))}
                    disabled={page === 1}
                    className="p-1.5 rounded-md border border-slate-200 bg-white text-slate-600 hover:bg-slate-50 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                  >
                    <CaretLeft size={16} weight="bold" />
                  </button>
                  <span className="text-sm font-medium text-slate-700 w-16 text-center">
                    Page {page}
                  </span>
                  <button
                    onClick={() => setPage(p => Math.min(totalPages, p + 1))}
                    disabled={page === totalPages}
                    className="p-1.5 rounded-md border border-slate-200 bg-white text-slate-600 hover:bg-slate-50 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                  >
                    <CaretRight size={16} weight="bold" />
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default DatabasePapersPage;
