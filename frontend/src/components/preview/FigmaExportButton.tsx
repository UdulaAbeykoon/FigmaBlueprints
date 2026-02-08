import React, { useState } from 'react';
import { Button } from '../ui/button';
import { FaFigma, FaCopy } from 'react-icons/fa';
import {
    Popover,
    PopoverContent,
    PopoverTrigger,
} from "../ui/popover";
import { Label } from "../ui/label";
import { Input } from "../ui/input";

interface Props {
    previewIframeRef: React.RefObject<HTMLIFrameElement>;
}

interface FigmaLayer {
    type: string;
    name?: string;
    x: number;
    y: number;
    width: number;
    height: number;
    backgroundColor?: string;
    color?: string;
    characters?: string;
    fontSize?: number;
    fontFamily?: string;
    fontWeight?: number;
    opacity?: number;
    borderRadius?: number;
    borderColor?: string;
    borderWidth?: number;
}

export const FigmaExportButton: React.FC<Props> = ({ previewIframeRef }) => {
    const [loading, setLoading] = useState(false);
    const [transferId, setTransferId] = useState<string | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [isOpen, setIsOpen] = useState(false);

    // Custom DOM walker that extracts COMPUTED styles
    const extractLayers = (element: Element, offsetX = 0, offsetY = 0): FigmaLayer[] => {
        const layers: FigmaLayer[] = [];
        const win = element.ownerDocument.defaultView;
        if (!win) return layers;

        const walkElement = (el: Element) => {
            // Skip hidden elements, scripts, styles, etc.
            const tagName = el.tagName.toLowerCase();
            if (['script', 'style', 'link', 'meta', 'head', 'noscript'].includes(tagName)) {
                return;
            }

            const styles = win.getComputedStyle(el);
            const rect = el.getBoundingClientRect();

            // Skip if invisible
            if (styles.display === 'none' || styles.visibility === 'hidden') {
                return;
            }

            // Skip if zero size
            if (rect.width === 0 || rect.height === 0) {
                // Still process children for layout containers
                Array.from(el.children).forEach(walkElement);
                return;
            }

            const x = rect.left - offsetX;
            const y = rect.top - offsetY;
            const width = rect.width;
            const height = rect.height;

            // Get background color (skip transparent)
            const bgColor = styles.backgroundColor;
            const hasBg = bgColor && bgColor !== 'rgba(0, 0, 0, 0)' && bgColor !== 'transparent';

            // Get text content (only direct text, not from children)
            let textContent = '';
            el.childNodes.forEach(node => {
                if (node.nodeType === Node.TEXT_NODE) {
                    const text = node.textContent?.trim();
                    if (text) textContent += text + ' ';
                }
            });
            textContent = textContent.trim();

            // Get text color
            const textColor = styles.color;

            // Get font properties
            const fontSize = parseFloat(styles.fontSize);
            const fontFamily = styles.fontFamily;
            const fontWeight = parseInt(styles.fontWeight) || 400;

            // Get border radius
            const borderRadius = parseFloat(styles.borderRadius) || 0;

            // Get border
            const borderWidth = parseFloat(styles.borderWidth) || 0;
            const borderColor = styles.borderColor;

            // Get opacity
            const opacity = parseFloat(styles.opacity);

            // Create layer based on content type
            if (textContent && textContent.length > 0) {
                // TEXT layer
                layers.push({
                    type: 'TEXT',
                    name: textContent.substring(0, 30) || tagName,
                    x, y, width, height,
                    characters: textContent,
                    color: textColor,
                    fontSize: fontSize,
                    fontFamily: fontFamily,
                    fontWeight: fontWeight,
                    opacity: opacity < 1 ? opacity : undefined
                });
            }

            if (hasBg || borderWidth > 0) {
                // RECTANGLE layer for backgrounds
                layers.push({
                    type: 'RECTANGLE',
                    name: el.className?.toString().split(' ')[0] || tagName,
                    x, y, width, height,
                    backgroundColor: hasBg ? bgColor : undefined,
                    borderRadius: borderRadius > 0 ? borderRadius : undefined,
                    borderColor: borderWidth > 0 ? borderColor : undefined,
                    borderWidth: borderWidth > 0 ? borderWidth : undefined,
                    opacity: opacity < 1 ? opacity : undefined
                });
            }

            // Handle images
            if (tagName === 'img' || tagName === 'svg') {
                layers.push({
                    type: 'IMAGE',
                    name: (el as HTMLImageElement).alt || 'Image',
                    x, y, width, height,
                    backgroundColor: '#cccccc' // Placeholder
                });
            }

            // Recursively process children
            Array.from(el.children).forEach(walkElement);
        };

        walkElement(element);
        return layers;
    };

    const handleExport = async () => {
        if (!previewIframeRef.current || !previewIframeRef.current.contentDocument) {
            alert("Preview not ready");
            return;
        }

        setLoading(true);
        setError(null);
        setTransferId(null);

        try {
            const iframeDoc = previewIframeRef.current.contentDocument;
            const iframeBody = iframeDoc.body;

            if (!iframeBody) {
                throw new Error("No body in iframe");
            }

            // Get iframe body rect for offset calculation
            const bodyRect = iframeBody.getBoundingClientRect();
            const offsetX = bodyRect.left;
            const offsetY = bodyRect.top;

            console.log("=== CUSTOM DOM WALKER ===");
            console.log("Body size:", bodyRect.width, "x", bodyRect.height);

            // Extract layers using computed styles directly from iframe
            const layers = extractLayers(iframeBody, offsetX, offsetY);

            console.log("Extracted layers:", layers.length);
            if (layers.length > 0) {
                console.log("First few layers:", layers.slice(0, 5));
            }

            // Add root container as first layer
            const rootLayer: FigmaLayer = {
                type: 'FRAME',
                name: 'Root',
                x: 0,
                y: 0,
                width: bodyRect.width || 1440,
                height: bodyRect.height || 900,
                backgroundColor: window.getComputedStyle(iframeBody).backgroundColor || '#ffffff'
            };

            const allLayers = [rootLayer, ...layers];

            // Send to backend
            const response = await fetch('http://localhost:7002/api/figma/upload', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ layers: allLayers }),
            });

            if (!response.ok) {
                throw new Error("Failed to upload design");
            }

            const data = await response.json();
            setTransferId(data.transferId);
            setIsOpen(true);

        } catch (error) {
            console.error("Figma Export Error:", error);
            if (error instanceof Error) {
                setError(`Export failed: ${error.message}`);
            } else {
                setError("Export failed with an unknown error.");
            }
            setIsOpen(true);
        } finally {
            setLoading(false);
        }
    };

    const copyToClipboard = () => {
        if (transferId) {
            navigator.clipboard.writeText(transferId);
        }
    };

    return (
        <Popover open={isOpen} onOpenChange={setIsOpen}>
            <PopoverTrigger asChild>
                <div className="flex items-center gap-2">
                    <Button
                        onClick={handleExport}
                        disabled={loading}
                        className="flex items-center gap-2 bg-[#F2C94C] text-black hover:bg-[#F2994A]"
                        title="Export to Figma"
                    >
                        <FaFigma />
                        {loading ? "Exporting..." : "Export to Figma"}
                    </Button>
                </div>
            </PopoverTrigger>
            <PopoverContent className="w-80">
                <div className="grid gap-4">
                    <div className="space-y-2">
                        <h4 className="font-medium leading-none">Export to Figma</h4>
                        <p className="text-sm text-muted-foreground">
                            {error ? "An error occurred." : "Enter this ID in the Figma plugin."}
                        </p>
                    </div>
                    {error ? (
                        <div className="text-sm text-red-500">{error}</div>
                    ) : (
                        <div className="flex items-center gap-2">
                            <div className="grid flex-1 gap-2">
                                <Label htmlFor="transferId" className="sr-only">
                                    Transfer ID
                                </Label>
                                <Input
                                    id="transferId"
                                    defaultValue={transferId || ""}
                                    readOnly
                                    className="h-8"
                                />
                            </div>
                            <Button size="sm" className="px-3" onClick={copyToClipboard}>
                                <span className="sr-only">Copy</span>
                                <FaCopy className="h-4 w-4" />
                            </Button>
                        </div>
                    )}
                </div>
            </PopoverContent>
        </Popover>
    );
};
