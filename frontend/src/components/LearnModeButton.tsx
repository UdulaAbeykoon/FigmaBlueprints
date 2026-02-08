/**
 * LearnModeButton Component
 * 
 * A button that enables "Learn Mode" for teaching users how to recreate
 * the generated design in Figma step-by-step.
 */



interface LearnModeButtonProps {
    onClick: () => void;
    disabled?: boolean;
}

export function LearnModeButton({ onClick, disabled }: LearnModeButtonProps) {
    return (
        <button
            onClick={onClick}
            disabled={disabled}
            className="inline-flex items-center gap-2 px-4 py-2 bg-[#3C437B] text-white rounded-lg font-medium hover:bg-[#3C437B]/90 transition-all disabled:opacity-50 disabled:cursor-not-allowed shadow-lg shadow-[#3C437B]/25"
            title="Learn how to recreate this design in Figma"
        >
            <svg
                className="w-5 h-5"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
            >
                <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253"
                />
            </svg>
            Learn in Figma
        </button>
    );
}

export default LearnModeButton;
