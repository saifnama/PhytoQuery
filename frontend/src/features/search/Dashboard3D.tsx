/**
 * Dashboard3D — 3D interactive knowledge graph showing entity interconnectiveness
 * across the entire corpus. Accessible from Dashboard via "3D View" button.
 */

import React, { useEffect, useState } from 'react';
import { ArrowLeft } from '@phosphor-icons/react';
import { useNavigate } from 'react-router-dom';
import { ForceGraph3D } from './ForceGraph3D';
import { dashboardApi } from '../../lib/api';
import type { Graph3DData } from '../../types';

const Dashboard3D: React.FC = () => {
  const [data, setData] = useState<Graph3DData | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const navigate = useNavigate();

  useEffect(() => {
    dashboardApi.getGraph3D()
      .then(setData)
      .catch((err) => setError(String(err)))
      .finally(() => setIsLoading(false));
  }, []);

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="flex items-center gap-4 px-6 py-4 border-b border-slate-200 bg-white">
        <button
          onClick={() => navigate('/')}
          className="flex items-center gap-2 text-sm text-slate-600 hover:text-slate-900 transition-colors"
        >
          <ArrowLeft size={16} weight="bold" />
          Back to Dashboard
        </button>
        <div className="h-4 w-px bg-slate-200" />
        <div>
          <h1 className="text-base font-semibold text-slate-900">3D Knowledge Graph</h1>
          <p className="text-xs text-slate-500">Entity interconnectiveness across all papers</p>
        </div>
        {data && (
          <div className="ml-auto text-xs text-slate-400">
            {data.nodes.length} entities · {data.links.length} connections
          </div>
        )}
      </div>

      {/* Graph */}
      <div className="flex-1 relative">
        {isLoading && (
          <div className="flex items-center justify-center h-full">
            <div className="flex items-center gap-3 text-slate-500 text-sm">
              <div className="w-5 h-5 border-2 border-blue-200 border-t-blue-600 rounded-full animate-spin" />
              Building 3D knowledge graph...
            </div>
          </div>
        )}
        {error && (
          <div className="flex items-center justify-center h-full text-red-500 text-sm">
            Failed to load graph: {error}
          </div>
        )}
        {data && !isLoading && (
          <div className="w-full h-full bg-slate-950" style={{ minHeight: 'calc(100vh - 80px)' }}>
            <ForceGraph3D data={data} />
          </div>
        )}
      </div>
    </div>
  );
};

export default Dashboard3D;