import { api } from './client';

// Search Types API
export const searchTypesApi = {
  /**
   * Fetch available article types for a search source.
   * @param source - "europepmc" or "openalex"
   */
  async getTypes(source: string): Promise<{ types: { key: string; display_name: string; count: number | null }[] }> {
    const response = await api.get('/search/types', { params: { source } });
    return response.data;
  },
};


// Dashboard API
export const dashboardApi = {
  async getMetrics() {
    const response = await api.get('/api/dashboard/metrics');
    return response.data;
  },
};

