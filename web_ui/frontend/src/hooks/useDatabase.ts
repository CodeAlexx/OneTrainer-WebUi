import { useState, useEffect, useCallback } from 'react';
import { databaseApi, DbStatus } from '../lib/api';

/**
 * Hook to check database status and provide database-enabled state.
 * 
 * @param autoRefresh - If true, refresh status on mount
 * @returns Database status and utility functions
 */
export function useDatabase(autoRefresh = true) {
    const [dbStatus, setDbStatus] = useState<DbStatus | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    const refresh = useCallback(async () => {
        setLoading(true);
        setError(null);
        try {
            const res = await databaseApi.getStatus();
            setDbStatus(res.data);
        } catch (err: any) {
            setError(err.message || 'Failed to get database status');
            setDbStatus(null);
        } finally {
            setLoading(false);
        }
    }, []);

    useEffect(() => {
        if (autoRefresh) {
            refresh();
        }
    }, [autoRefresh, refresh]);

    // Database is usable if enabled and initialized
    const dbEnabled = dbStatus?.enabled && dbStatus?.initialized;

    return {
        dbStatus,
        dbEnabled: !!dbEnabled,
        loading,
        error,
        refresh,
    };
}

export default useDatabase;
