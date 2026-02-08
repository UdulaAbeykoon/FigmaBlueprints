/**
 * useLearnMode Hook - Manages Learn Mode state and integration
 */

import { useState, useCallback } from "react";

interface LearnModeState {
    isActive: boolean;
    htmlCode: string | null;
}

export function useLearnMode() {
    const [state, setState] = useState<LearnModeState>({
        isActive: false,
        htmlCode: null,
    });

    const startLearnMode = useCallback((htmlCode: string) => {
        setState({
            isActive: true,
            htmlCode,
        });
    }, []);

    const stopLearnMode = useCallback(() => {
        setState({
            isActive: false,
            htmlCode: null,
        });
    }, []);

    return {
        isLearnModeActive: state.isActive,
        learnModeHtmlCode: state.htmlCode,
        startLearnMode,
        stopLearnMode,
    };
}
