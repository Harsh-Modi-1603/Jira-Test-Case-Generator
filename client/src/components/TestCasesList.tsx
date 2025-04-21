import { useState } from "react";
import {
    Card,
    CardContent,
    CardFooter,
    CardHeader,
    CardTitle,
} from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import {
    Check,
    Copy,
    Download,
    FilePlus,
    X,
    Tag,
    AlertCircle,
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { toast } from "sonner";
import { ParsedTestCase } from "@/lib/testCaseParser";
import {
    Collapsible,
    CollapsibleContent,
    CollapsibleTrigger,
} from "@/components/ui/collapsible";

interface TestCasesListProps {
    storyId: string;
    testCases: ParsedTestCase[];
    rawContent?: string;
    onClose: () => void;
}

export const TestCasesList = ({
    storyId,
    testCases,
    rawContent,
    onClose,
}: TestCasesListProps) => {
    const [selectedTab, setSelectedTab] = useState<string>("all");
    const [copiedId, setCopiedId] = useState<string | null>(null);
    const [expandedScenarios, setExpandedScenarios] = useState<Set<string>>(
        new Set()
    );

    // Group test cases by scenario
    const groupedTestCases = testCases.reduce((acc, tc) => {
        const scenarioId = tc.scenarioId || "ungrouped";
        if (!acc[scenarioId]) {
            acc[scenarioId] = {
                id: scenarioId,
                title: tc.scenario || "Ungrouped Test Cases",
                testCases: [],
            };
        }
        acc[scenarioId].testCases.push(tc);
        return acc;
    }, {} as Record<string, { id: string; title: string; testCases: ParsedTestCase[] }>);

    // Create array of scenarios
    const scenarios = Object.values(groupedTestCases).sort((a, b) =>
        a.id.localeCompare(b.id)
    );

    const toggleScenario = (scenarioId: string) => {
        const newExpanded = new Set(expandedScenarios);
        if (newExpanded.has(scenarioId)) {
            newExpanded.delete(scenarioId);
        } else {
            newExpanded.add(scenarioId);
        }
        setExpandedScenarios(newExpanded);
    };

    const handleCopyToClipboard = (text: string, id: string) => {
        navigator.clipboard.writeText(text);
        setCopiedId(id);

        setTimeout(() => {
            setCopiedId(null);
        }, 2000);

        toast.success("Copied to clipboard");
    };

    const handleDownload = () => {
        const content = testCases
            .map((tc) => {
                return `
TEST CASE ID: ${tc.id}
TEST CASE: ${tc.title}
${tc.preconditions ? `PRECONDITIONS: ${tc.preconditions}\n` : ""}
${tc.testData ? `TEST DATA: ${tc.testData}\n` : ""}
STEPS:
${tc.steps.map((step, idx) => `  ${idx + 1}. ${step}`).join("\n")}
EXPECTED RESULT:
  ${tc.expectedResult}
${tc.priority ? `PRIORITY: ${tc.priority}\n` : ""}
${tc.references ? `REFERENCES: ${tc.references}\n` : ""}
-------------------
`;
            })
            .join("\n");

        const blob = new Blob([content], { type: "text/plain" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `${storyId}-test-cases.txt`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);

        toast.success("Test cases downloaded");
    };

    const handleCopyRawContent = () => {
        if (rawContent) {
            navigator.clipboard.writeText(rawContent);
            toast.success("Raw content copied to clipboard");
        }
    };

    const handleDownloadRawContent = () => {
        if (!rawContent) return;

        const blob = new Blob([rawContent], { type: "text/plain" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `${storyId}-raw-content.txt`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);

        toast.success("Raw content downloaded");
    };

    // Function to get priority badge color
    const getPriorityColor = (priority?: string) => {
        switch (priority?.toLowerCase()) {
            case "high":
                return "bg-jira-red text-white";
            case "medium":
                return "bg-jira-yellow text-black";
            case "low":
                return "bg-jira-green text-white";
            default:
                return "bg-gray-500 text-white";
        }
    };

    return (
        <Card className="w-full max-w-3xl mx-auto animate-scale-in shadow-lg">
            <CardHeader className="pb-3 flex flex-row items-center justify-between">
                <div>
                    <div className="flex items-center gap-2 mb-1">
                        <Badge variant="outline" className="font-mono">
                            {storyId}
                        </Badge>
                        <Badge className="bg-green-500 text-white">
                            {testCases.length} Test Cases
                        </Badge>
                    </div>
                    <CardTitle className="text-xl">
                        Generated Test Cases
                    </CardTitle>
                </div>
                <Button variant="ghost" size="icon" onClick={onClose}>
                    <X className="h-4 w-4" />
                </Button>
            </CardHeader>

            <Separator />

            <Tabs
                defaultValue="all"
                onValueChange={setSelectedTab}
                className="w-full"
            >
                <div className="px-6 pt-3">
                    <TabsList className="grid grid-cols-2 mb-4 w-full">
                        <TabsTrigger value="all" className="text-xs">
                            All ({testCases.length})
                        </TabsTrigger>
                        <TabsTrigger value="raw" className="text-xs">
                            Raw
                        </TabsTrigger>
                    </TabsList>
                </div>

                <CardContent className="pt-0 pb-0">
                    <TabsContent value="all" className="mt-0">
                        <ScrollArea className="h-[60vh] pr-4">
                            <div className="space-y-4">
                                {scenarios.map((scenario) => (
                                    <Collapsible
                                        key={scenario.id}
                                        open={expandedScenarios.has(
                                            scenario.id
                                        )}
                                        onOpenChange={() =>
                                            toggleScenario(scenario.id)
                                        }
                                    >
                                        <Card className="border-l-4 border-l-blue-500">
                                            <CollapsibleTrigger asChild>
                                                <CardHeader className="p-4 cursor-pointer hover:bg-slate-50 dark:hover:bg-slate-900">
                                                    <div className="flex items-center justify-between">
                                                        <div>
                                                            <Badge
                                                                variant="outline"
                                                                className="font-mono mb-2"
                                                            >
                                                                {scenario.id}
                                                            </Badge>
                                                            <h3 className="font-medium">
                                                                {scenario.title}
                                                            </h3>
                                                        </div>
                                                        <Badge>
                                                            {
                                                                scenario
                                                                    .testCases
                                                                    .length
                                                            }{" "}
                                                            Tests
                                                        </Badge>
                                                    </div>
                                                </CardHeader>
                                            </CollapsibleTrigger>
                                            <CollapsibleContent>
                                                <CardContent className="pt-0 pb-4">
                                                    <div className="space-y-4 pl-2">
                                                        {scenario.testCases.map(
                                                            (testCase) => (
                                                                <Card
                                                                    key={
                                                                        testCase.id
                                                                    }
                                                                    className="border-l-4 border-l-primary"
                                                                >
                                                                    <CardHeader className="pb-2">
                                                                        <div className="flex items-center justify-between">
                                                                            <div className="flex items-center gap-2">
                                                                                <Badge
                                                                                    variant="outline"
                                                                                    className="font-mono text-xs"
                                                                                >
                                                                                    {
                                                                                        testCase.id
                                                                                    }
                                                                                </Badge>
                                                                                {testCase.priority && (
                                                                                    <Badge
                                                                                        className={getPriorityColor(
                                                                                            testCase.priority
                                                                                        )}
                                                                                    >
                                                                                        {
                                                                                            testCase.priority
                                                                                        }
                                                                                    </Badge>
                                                                                )}
                                                                            </div>
                                                                            <Button
                                                                                variant="ghost"
                                                                                size="icon"
                                                                                className="h-8 w-8"
                                                                                onClick={() => {
                                                                                    const text = `
TEST CASE ID: ${testCase.id}
TEST CASE: ${testCase.title}
${testCase.preconditions ? `PRECONDITIONS: ${testCase.preconditions}\n` : ""}
${testCase.testData ? `TEST DATA: ${testCase.testData}\n` : ""}
STEPS:
${testCase.steps.map((step, idx) => `${idx + 1}. ${step}`).join("\n")}
EXPECTED RESULT:
${testCase.expectedResult}
${testCase.priority ? `PRIORITY: ${testCase.priority}` : ""}
${testCase.references ? `REFERENCES: ${testCase.references}` : ""}`;
                                                                                    handleCopyToClipboard(
                                                                                        text,
                                                                                        testCase.id
                                                                                    );
                                                                                }}
                                                                            >
                                                                                {copiedId ===
                                                                                testCase.id ? (
                                                                                    <Check className="h-4 w-4 text-green-500" />
                                                                                ) : (
                                                                                    <Copy className="h-4 w-4" />
                                                                                )}
                                                                            </Button>
                                                                        </div>
                                                                        <h3 className="font-medium text-sm">
                                                                            {
                                                                                testCase.title
                                                                            }
                                                                        </h3>
                                                                    </CardHeader>
                                                                    <CardContent className="pb-2 text-sm">
                                                                        {testCase.preconditions && (
                                                                            <div className="mb-3 space-y-1">
                                                                                <h4 className="font-medium text-xs text-muted-foreground">
                                                                                    Preconditions
                                                                                </h4>
                                                                                <p className="text-muted-foreground">
                                                                                    {
                                                                                        testCase.preconditions
                                                                                    }
                                                                                </p>
                                                                            </div>
                                                                        )}

                                                                        {testCase.testData &&
                                                                            testCase.testData !==
                                                                                "N/A" && (
                                                                                <div className="mb-3 space-y-1">
                                                                                    <h4 className="font-medium text-xs text-muted-foreground">
                                                                                        Test
                                                                                        Data
                                                                                    </h4>
                                                                                    <p className="text-muted-foreground">
                                                                                        {
                                                                                            testCase.testData
                                                                                        }
                                                                                    </p>
                                                                                </div>
                                                                            )}

                                                                        <div className="space-y-2">
                                                                            <h4 className="font-medium text-xs text-muted-foreground">
                                                                                Steps
                                                                            </h4>
                                                                            <ol className="list-decimal list-inside space-y-1">
                                                                                {testCase.steps.map(
                                                                                    (
                                                                                        step,
                                                                                        idx
                                                                                    ) => (
                                                                                        <li
                                                                                            key={
                                                                                                idx
                                                                                            }
                                                                                            className="text-muted-foreground"
                                                                                        >
                                                                                            {
                                                                                                step
                                                                                            }
                                                                                        </li>
                                                                                    )
                                                                                )}
                                                                            </ol>
                                                                        </div>

                                                                        <div className="mt-3 space-y-1">
                                                                            <h4 className="font-medium text-xs text-muted-foreground">
                                                                                Expected
                                                                                Result
                                                                            </h4>
                                                                            <p className="text-muted-foreground">
                                                                                {
                                                                                    testCase.expectedResult
                                                                                }
                                                                            </p>
                                                                        </div>

                                                                        {(testCase.passCriteria ||
                                                                            testCase.failCriteria) && (
                                                                            <div className="mt-3 space-y-1">
                                                                                <h4 className="font-medium text-xs text-muted-foreground">
                                                                                    Pass/Fail
                                                                                    Criteria
                                                                                </h4>
                                                                                {testCase.passCriteria && (
                                                                                    <div className="flex gap-2 items-start">
                                                                                        <Check className="h-4 w-4 text-green-500 mt-0.5" />
                                                                                        <p className="text-muted-foreground">
                                                                                            {
                                                                                                testCase.passCriteria
                                                                                            }
                                                                                        </p>
                                                                                    </div>
                                                                                )}
                                                                                {testCase.failCriteria && (
                                                                                    <div className="flex gap-2 items-start">
                                                                                        <X className="h-4 w-4 text-red-500 mt-0.5" />
                                                                                        <p className="text-muted-foreground">
                                                                                            {
                                                                                                testCase.failCriteria
                                                                                            }
                                                                                        </p>
                                                                                    </div>
                                                                                )}
                                                                            </div>
                                                                        )}

                                                                        {testCase.references && (
                                                                            <div className="mt-3 flex items-center gap-1">
                                                                                <Tag
                                                                                    size={
                                                                                        12
                                                                                    }
                                                                                    className="text-purple-500"
                                                                                />
                                                                                <span className="text-xs text-purple-500">
                                                                                    {
                                                                                        testCase.references
                                                                                    }
                                                                                </span>
                                                                            </div>
                                                                        )}
                                                                    </CardContent>
                                                                </Card>
                                                            )
                                                        )}
                                                    </div>
                                                </CardContent>
                                            </CollapsibleContent>
                                        </Card>
                                    </Collapsible>
                                ))}
                            </div>
                        </ScrollArea>
                    </TabsContent>
                    <TabsContent value="raw" className="mt-0">
                        <ScrollArea className="h-[60vh] pr-4">
                            <div className="space-y-4">
                                <Card>
                                    <CardHeader className="pb-2">
                                        <div className="flex items-center justify-between">
                                            <h3 className="font-medium">
                                                Raw AI Response
                                            </h3>
                                            <div className="flex gap-2">
                                                <Button
                                                    variant="outline"
                                                    size="sm"
                                                    className="h-8"
                                                    onClick={
                                                        handleCopyRawContent
                                                    }
                                                >
                                                    <Copy className="h-4 w-4 mr-1" />{" "}
                                                    Copy
                                                </Button>
                                                <Button
                                                    variant="outline"
                                                    size="sm"
                                                    className="h-8"
                                                    onClick={
                                                        handleDownloadRawContent
                                                    }
                                                >
                                                    <Download className="h-4 w-4 mr-1" />{" "}
                                                    Download
                                                </Button>
                                            </div>
                                        </div>
                                    </CardHeader>
                                    <CardContent className="pb-2">
                                        {rawContent ? (
                                            <pre className="whitespace-pre-wrap text-sm font-mono bg-slate-50 dark:bg-slate-900 p-4 rounded-md overflow-auto">
                                                {rawContent}
                                            </pre>
                                        ) : (
                                            <p className="text-muted-foreground text-center p-4">
                                                No raw content available.
                                            </p>
                                        )}
                                    </CardContent>
                                </Card>
                            </div>
                        </ScrollArea>
                    </TabsContent>
                </CardContent>
            </Tabs>

            <Separator className="mt-4" />

            <CardFooter className="py-3 flex justify-between">
                <Button
                    variant="outline"
                    onClick={handleDownload}
                    className="gap-1.5"
                >
                    <Download className="h-4 w-4" />
                    <span>Download Tests</span>
                </Button>
                <Button variant="outline" onClick={onClose} className="gap-1.5">
                    <FilePlus className="h-4 w-4" />
                    <span>Return to Stories</span>
                </Button>
            </CardFooter>
        </Card>
    );
};

// eslint-disable-next-line @typescript-eslint/no-explicit-any
const cn = (...classes: any[]) => classes.filter(Boolean).join(" ");
