/**
 * TutorOverlay Component - Learn Mode Screen Capture and Guidance
 * 
 * This component provides a floating "Tutor Card" that:
 * 1. Captures the user's screen (targeting the Figma window)
 * 2. Displays step-by-step instructions
 * 3. Verify progress on demand using AI vision
 */

import { useState, useEffect, useRef, useCallback, MouseEvent as ReactMouseEvent } from "react";

// Types
interface LessonStep {
    id: number;
    instruction: string;
    success_criteria: string;
}

interface LessonPlan {
    steps: LessonStep[];
    total_steps: number;
    estimated_time_minutes: number;
}

interface VerifyResponse {
    completed: boolean;
    feedback: string;
    confidence: number;
}

interface TutorOverlayProps {
    htmlCode: string;
    onClose: () => void;
    backendUrl?: string;
}

// API Functions
const API_BASE = "http://localhost:7002";

async function generateLessonPlan(htmlCode: string): Promise<LessonPlan> {
    const response = await fetch(`${API_BASE}/api/learn/generate-lesson-plan`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ html_code: htmlCode, framework: "tailwind" }),
    });

    if (!response.ok) {
        throw new Error(`Failed to generate lesson plan: ${response.statusText}`);
    }

    return response.json();
}

async function verifyProgress(
    step: LessonStep,
    screenshotBase64: string
): Promise<VerifyResponse> {
    const response = await fetch(`${API_BASE}/api/learn/verify-progress`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            current_step: step,
            screenshot_base64: screenshotBase64,
        }),
    });

    if (!response.ok) {
        throw new Error(`Failed to verify progress: ${response.statusText}`);
    }

    return response.json();
}

// Screen Capture Hook
function useScreenCapture() {
    const [stream, setStream] = useState<MediaStream | null>(null);
    const [isCapturing, setIsCapturing] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const videoRef = useRef<HTMLVideoElement | null>(null);
    const canvasRef = useRef<HTMLCanvasElement | null>(null);

    const startCapture = useCallback(async () => {
        try {
            setError(null);
            const mediaStream = await navigator.mediaDevices.getDisplayMedia({
                video: {
                    displaySurface: "window",
                    // @ts-ignore - Chrome specific
                    preferCurrentTab: false,
                },
                audio: false,
            });

            setStream(mediaStream);
            setIsCapturing(true);

            // Handle stream ending (user clicks "Stop sharing")
            mediaStream.getVideoTracks()[0].onended = () => {
                setIsCapturing(false);
                setStream(null);
            };
        } catch (err) {
            setError("Screen capture cancelled or not supported.");
            setIsCapturing(false);
        }
    }, []);

    const stopCapture = useCallback(() => {
        if (stream) {
            stream.getTracks().forEach((track) => track.stop());
            setStream(null);
        }
        setIsCapturing(false);
    }, [stream]);

    const captureFrame = useCallback((): string | null => {
        if (!videoRef.current || !canvasRef.current || !isCapturing) {
            return null;
        }

        const video = videoRef.current;
        const canvas = canvasRef.current;
        const ctx = canvas.getContext("2d");

        if (!ctx || video.videoWidth === 0) {
            return null;
        }

        // Set canvas size to match video but limit max dimension
        // Groq API has 4MB limit for base64
        const MAX_WIDTH = 1280;
        let width = video.videoWidth;
        let height = video.videoHeight;

        if (width > MAX_WIDTH) {
            const scale = MAX_WIDTH / width;
            width = MAX_WIDTH;
            height = height * scale;
        }

        canvas.width = width;
        canvas.height = height;

        // Draw current frame scaled
        ctx.drawImage(video, 0, 0, width, height);

        // Convert to base64 (send FULL string for backend processing)
        // Using image/jpeg for better compression/speed
        return canvas.toDataURL("image/jpeg", 0.7);
    }, [isCapturing]);

    return {
        stream,
        isCapturing,
        error,
        startCapture,
        stopCapture,
        captureFrame,
        videoRef,
        canvasRef,
    };
}

// Main Component
export function TutorOverlay({ htmlCode, onClose }: TutorOverlayProps) {
    // Lesson State
    const [lessonPlan, setLessonPlan] = useState<LessonPlan | null>(null);
    const [currentStepIndex, setCurrentStepIndex] = useState(0);
    const [isLoading, setIsLoading] = useState(true);
    const [loadError, setLoadError] = useState<string | null>(null);

    // Verification State
    const [isVerifying, setIsVerifying] = useState(false);
    const [feedback, setFeedback] = useState<string>("");
    const [messageType, setMessageType] = useState<"info" | "success" | "error">("info");

    // Screen Capture
    const {
        stream,
        isCapturing,
        error: captureError,
        startCapture,
        stopCapture,
        captureFrame,
        videoRef,
        canvasRef,
    } = useScreenCapture();

    // Draggable Position
    const [position, setPosition] = useState({ x: 20, y: 20 });
    const [isDragging, setIsDragging] = useState(false);
    const dragOffset = useRef({ x: 0, y: 0 });

    // Privacy Acknowledged
    const [privacyAcknowledged, setPrivacyAcknowledged] = useState(false);

    // Load lesson plan on mount
    useEffect(() => {
        async function loadLesson() {
            try {
                setIsLoading(true);
                const plan = await generateLessonPlan(htmlCode);
                setLessonPlan(plan);
                setLoadError(null);
            } catch (err) {
                setLoadError(err instanceof Error ? err.message : "Failed to load lesson");
            } finally {
                setIsLoading(false);
            }
        }

        loadLesson();
    }, [htmlCode]);

    // Attach video stream to video element
    useEffect(() => {
        if (videoRef.current && stream) {
            videoRef.current.srcObject = stream;
            videoRef.current.play();
        }
    }, [stream, videoRef]);

    // Manual Verification Handler
    const handleCheckProgress = async () => {
        if (!isCapturing || !lessonPlan) return;

        const currentStep = lessonPlan.steps[currentStepIndex];
        const screenshot = captureFrame();

        if (!screenshot) {
            setFeedback("Could not capture screen. Make sure visual sharing is active.");
            setMessageType("error");
            return;
        }

        setIsVerifying(true);
        setFeedback("Analyzing your screen...");
        setMessageType("info");

        try {
            const result = await verifyProgress(currentStep, screenshot);

            if (result.completed && result.confidence >= 0.5) {
                setFeedback("🎉 Success! Moving to next step...");
                setMessageType("success");

                // Wait 1.5s then advance
                setTimeout(() => {
                    if (currentStepIndex < lessonPlan.steps.length - 1) {
                        setCurrentStepIndex(prev => prev + 1);
                        setFeedback("");
                        setMessageType("info");
                    } else {
                        setFeedback("🏆 All steps completed! Great job!");
                        setMessageType("success");
                    }
                }, 1500);

            } else {
                setFeedback(result.feedback || "Look closer, something's missing.");
                setMessageType("error"); // Use error style for "not quite there"
            }
        } catch (err) {
            console.error("Verification error:", err);
            setFeedback("Connection error. Please try again.");
            setMessageType("error");
        } finally {
            setIsVerifying(false);
        }
    };

    // Dragging handlers
    const handleMouseDown = (e: ReactMouseEvent) => {
        if ((e.target as HTMLElement).closest(".drag-handle")) {
            setIsDragging(true);
            dragOffset.current = {
                x: e.clientX - position.x,
                y: e.clientY - position.y,
            };
        }
    };

    const handleMouseMove = useCallback(
        (e: MouseEvent) => {
            if (isDragging) {
                setPosition({
                    x: e.clientX - dragOffset.current.x,
                    y: e.clientY - dragOffset.current.y,
                });
            }
        },
        [isDragging]
    );

    const handleMouseUp = useCallback(() => {
        setIsDragging(false);
    }, []);

    useEffect(() => {
        if (isDragging) {
            window.addEventListener("mousemove", handleMouseMove);
            window.addEventListener("mouseup", handleMouseUp);
        }
        return () => {
            window.removeEventListener("mousemove", handleMouseMove);
            window.removeEventListener("mouseup", handleMouseUp);
        };
    }, [isDragging, handleMouseMove, handleMouseUp]);

    // Navigation
    const handleSkipStep = () => {
        if (lessonPlan && currentStepIndex < lessonPlan.steps.length - 1) {
            setCurrentStepIndex((prev) => prev + 1);
            setFeedback("");
            setMessageType("info");
        }
    };

    const handlePrevStep = () => {
        if (currentStepIndex > 0) {
            setCurrentStepIndex((prev) => prev - 1);
            setFeedback("");
            setMessageType("info");
        }
    };

    const currentStep = lessonPlan?.steps[currentStepIndex];
    const progress = lessonPlan
        ? ((currentStepIndex + 1) / lessonPlan.total_steps) * 100
        : 0;

    // Privacy Warning Screen
    if (!privacyAcknowledged) {
        return (
            <div
                className="fixed inset-0 bg-black/50 flex items-center justify-center z-[9999]"
                style={{ backdropFilter: "blur(4px)" }}
            >
                <div className="bg-white/90 backdrop-blur-xl border border-white/60 rounded-2xl p-8 max-w-md mx-4 shadow-2xl">
                    <div className="text-center">
                        <div className="flex justify-center mb-6">
                            <img
                                src="/favicon/Schema%20Logo.png"
                                alt="Schema Logo"
                                className="w-32 object-contain"
                            />
                        </div>
                        <h2 className="text-xl font-bold text-gray-900 mb-2">
                            Screen Sharing Required
                        </h2>
                        <p className="text-gray-600 mb-6">
                            <strong>Privacy Notice:</strong> We will analyze your screen to
                            guide you through the Figma tutorial. Screenshots are processed
                            temporarily and not stored.
                        </p>
                        <div className="flex gap-3">
                            <button
                                onClick={onClose}
                                className="flex-1 px-4 py-2 border border-gray-300 rounded-lg text-gray-700 hover:bg-gray-50 transition-colors"
                            >
                                Cancel
                            </button>
                            <button
                                onClick={() => setPrivacyAcknowledged(true)}
                                className="flex-1 px-4 py-2 bg-[#3C437B] text-white rounded-lg hover:bg-[#3C437B]/90 transition-colors"
                            >
                                I Understand
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        );
    }

    return (
        <>
            {/* Hidden video and canvas for screen capture */}
            <video ref={videoRef} className="hidden" muted playsInline />
            <canvas ref={canvasRef} className="hidden" />

            {/* Backdrop */}
            <div className="fixed inset-0 bg-black/30 backdrop-blur-sm z-[9998]" />

            {/* Floating Tutor Card */}
            <div
                className="fixed z-[9999] bg-white dark:bg-gray-800 rounded-2xl shadow-2xl border border-gray-200 dark:border-gray-700 font-sans"
                style={{
                    left: position.x,
                    top: position.y,
                    width: 380,
                    userSelect: isDragging ? "none" : "auto",
                }}
                onMouseDown={handleMouseDown}
            >
                {/* Header - Drag Handle */}
                <div className="drag-handle flex items-center justify-between px-4 py-3 bg-gradient-to-r from-[#3C437B] to-purple-600 rounded-t-2xl cursor-move">
                    <div className="flex items-center gap-2">
                        <div className="w-2 h-2 rounded-full bg-white/30" />
                        <span className="text-white font-semibold">Tutor Mode</span>
                    </div>
                    <button
                        onClick={() => {
                            stopCapture();
                            onClose();
                        }}
                        className="text-white/70 hover:text-white"
                    >
                        <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                        </svg>
                    </button>
                </div>

                {/* Content */}
                <div className="p-4">
                    {isLoading ? (
                        <div className="text-center py-8">
                            <div className="animate-spin w-8 h-8 border-4 border-[#3C437B] border-t-transparent rounded-full mx-auto mb-3" />
                            <p className="text-gray-600 dark:text-gray-300">Generating lesson plan...</p>
                        </div>
                    ) : loadError ? (
                        <div className="text-center py-8">
                            <p className="text-red-500 mb-4">{loadError}</p>
                            <button
                                onClick={onClose}
                                className="px-4 py-2 bg-gray-200 dark:bg-gray-700 rounded-lg"
                            >
                                Close
                            </button>
                        </div>
                    ) : (
                        <>
                            {/* Progress Bar */}
                            <div className="mb-4">
                                <div className="flex justify-between text-sm text-gray-500 dark:text-gray-400 mb-1">
                                    <span>
                                        Step {currentStepIndex + 1} of {lessonPlan?.total_steps}
                                    </span>
                                    <span>{Math.round(progress)}%</span>
                                </div>
                                <div className="h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                                    <div
                                        className="h-full bg-gradient-to-r from-[#3C437B] to-purple-500 transition-all duration-500"
                                        style={{ width: `${progress}%` }}
                                    />
                                </div>
                            </div>

                            {/* Current Step */}
                            {currentStep && (
                                <div className="bg-gray-50 dark:bg-gray-700/50 rounded-xl p-4 mb-4">
                                    <p className="text-gray-900 dark:text-white font-medium mb-2 leading-relaxed">
                                        {currentStep.instruction}
                                    </p>
                                    <p className="text-sm text-gray-500 dark:text-gray-400 border-t border-gray-200 dark:border-gray-600 pt-2 mt-2">
                                        <span className="font-semibold text-[#3C437B]">Goal:</span> {currentStep.success_criteria}
                                    </p>
                                </div>
                            )}

                            {/* Feedback Message */}
                            {feedback && (
                                <div className={`border rounded-lg p-3 mb-4 text-sm font-medium transition-all ${messageType === 'success' ? 'bg-green-50 text-green-800 border-green-200' :
                                    messageType === 'error' ? 'bg-orange-50 text-orange-800 border-orange-200' :
                                        'bg-[#3C437B]/10 text-[#3C437B] border-[#3C437B]/20'
                                    }`}>
                                    {feedback}
                                </div>
                            )}

                            {/* Main Action Button */}
                            {!isCapturing ? (
                                <button
                                    onClick={startCapture}
                                    className="w-full py-3 bg-[#3C437B] text-white rounded-xl font-bold hover:bg-[#3C437B]/90 transition-all shadow-lg shadow-[#3C437B]/30 flex items-center justify-center gap-2 mb-4"
                                >
                                    <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
                                    </svg>
                                    Share Screen to Start
                                </button>
                            ) : (
                                <button
                                    onClick={handleCheckProgress}
                                    disabled={isVerifying}
                                    className={`w-full py-4 text-white rounded-xl font-bold text-lg shadow-lg transition-all transform active:scale-95 flex items-center justify-center gap-2 mb-4 ${isVerifying
                                        ? "bg-gray-400 cursor-not-allowed"
                                        : "bg-gradient-to-r from-green-500 to-emerald-600 hover:from-green-600 hover:to-emerald-700 shadow-green-500/30"
                                        }`}
                                >
                                    {isVerifying ? (
                                        <>
                                            <div className="animate-spin w-5 h-5 border-2 border-white border-t-transparent rounded-full" />
                                            Checking...
                                        </>
                                    ) : (
                                        <>
                                            <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M5 13l4 4L19 7" />
                                            </svg>
                                            Check My Work
                                        </>
                                    )}
                                </button>
                            )}

                            {captureError && (
                                <p className="text-sm text-red-500 mt-2 text-center mb-2">{captureError}</p>
                            )}

                            {/* Navigation Controls */}
                            <div className="flex items-center justify-between pt-2 border-t border-gray-200 dark:border-gray-700">
                                <button
                                    onClick={handlePrevStep}
                                    disabled={currentStepIndex === 0}
                                    className="text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-white disabled:opacity-30 text-sm font-medium px-2 py-1"
                                >
                                    ← Previous Step
                                </button>

                                <button
                                    onClick={handleSkipStep}
                                    disabled={!lessonPlan || currentStepIndex >= lessonPlan.steps.length - 1}
                                    className="text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-white disabled:opacity-30 text-sm font-medium px-2 py-1"
                                >
                                    Skip Step →
                                </button>
                            </div>
                        </>
                    )}
                </div>
            </div>
        </>
    );
}

export default TutorOverlay;
