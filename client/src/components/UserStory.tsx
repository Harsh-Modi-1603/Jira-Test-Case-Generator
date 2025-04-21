import { useState } from "react";
import {
    Card,
    CardContent,
    CardFooter,
    CardHeader,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import {
    FileText,
    Tag,
    Clock,
    Calendar,
    User,
    CheckSquare,
    PenLine,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { toast } from "sonner";
import { useGenerateTestCases } from "@/hooks/jira";

export interface UserStoryProps {
    id: string;
    title: string;
    description: string;
    priority: "Highest" | "High" | "Medium" | "Low" | "Lowest";
    status: "To Do" | "In Progress" | "Done" | "Blocked";
    assignee: string;
    dueDate: string;
    epicLink?: string;
    tags: string[];
    // Updated to handle AI response format
    onGenerateTests: (
        id: string,
        response: { content: string; token_count: number }
    ) => void;
}

const priorityColorMap = {
    Highest: "bg-jira-red text-white",
    High: "bg-jira-yellow text-black",
    Medium: "bg-jira-lightblue text-white",
    Low: "bg-jira-green text-white",
    Lowest: "bg-jira-teal text-white",
};

const statusColorMap = {
    "To Do": "bg-gray-200 text-gray-700",
    "In Progress": "bg-jira-blue text-white",
    Done: "bg-jira-green text-white",
    Blocked: "bg-jira-red text-white",
};

export const UserStory = ({
    id,
    title,
    description,
    priority,
    status,
    assignee,
    dueDate,
    epicLink,
    tags,
    onGenerateTests,
}: UserStoryProps) => {
    const [isExpanded, setIsExpanded] = useState(false);
    const [isGenerating, setIsGenerating] = useState(false);

    const { mutate: generateTests } = useGenerateTestCases(
        (data) => {
            setIsGenerating(false);
            toast.success(`Test cases generated for ${id}`);
            // Pass the raw AI response to parent component for parsing
            onGenerateTests(id, {
                content: data.content,
                token_count: data.token_count,
            });
        },
        () => {
            setIsGenerating(false);
            toast.error("Failed to generate test cases");
        }
    );

    const handleGenerateTests = () => {
        setIsGenerating(true);

        // Use the real test case generation endpoint
        generateTests({
            jiraId: id,
            userStory: `${title} - ${description}`,
            acceptanceCriteria: epicLink ? `Epic: ${epicLink}` : undefined,
        });
    };

    return (
        <Card
            className={cn(
                "w-full transition-all duration-300 hover:shadow-md overflow-hidden",
                "border-l-4",
                status === "In Progress" && "border-l-jira-blue",
                status === "Done" && "border-l-jira-green",
                status === "To Do" && "border-l-gray-300",
                status === "Blocked" && "border-l-jira-red"
            )}
        >
            <CardHeader className="pt-5 pb-3 flex flex-row items-start gap-4">
                <div className="flex-1">
                    <div className="flex items-center space-x-2 mb-2">
                        <Badge variant="outline" className="font-mono">
                            {id}
                        </Badge>
                        <Badge
                            className={cn("story-tag", statusColorMap[status])}
                        >
                            {status}
                        </Badge>
                        <Badge
                            className={cn(
                                "story-tag",
                                priorityColorMap[priority]
                            )}
                        >
                            {priority}
                        </Badge>
                    </div>
                    <h3 className="text-lg font-medium leading-6 text-balance">
                        {title}
                    </h3>
                </div>
            </CardHeader>

            <CardContent className="pt-0 pb-3 space-y-2">
                <div className="flex flex-wrap items-center gap-x-4 gap-y-2 text-sm">
                    <div className="flex items-center gap-1 text-muted-foreground">
                        <User size={14} />
                        <span>{assignee}</span>
                    </div>
                    <div className="flex items-center gap-1 text-muted-foreground">
                        <Calendar size={14} />
                        <span>{dueDate}</span>
                    </div>
                    {epicLink && (
                        <div className="flex items-center gap-1 text-purple-500">
                            <Tag size={14} />
                            <span>{epicLink}</span>
                        </div>
                    )}
                </div>

                <div className="flex flex-wrap gap-1 mt-2">
                    {tags.map((tag) => (
                        <Badge
                            key={tag}
                            variant="secondary"
                            className="text-xs py-0"
                        >
                            {tag}
                        </Badge>
                    ))}
                </div>

                <div
                    className={cn(
                        "mt-3 transition-all duration-300 overflow-hidden",
                        isExpanded ? "max-h-96" : "max-h-16"
                    )}
                >
                    <p className="text-sm text-muted-foreground text-pretty">
                        {description}
                    </p>
                </div>

                <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => setIsExpanded(!isExpanded)}
                    className="mt-2 p-0 h-auto text-muted-foreground hover:bg-transparent hover:underline"
                >
                    {isExpanded ? "Show less" : "Show more"}
                </Button>
            </CardContent>

            <Separator />

            <CardFooter className="py-3 flex justify-between">
                <div className="flex items-center gap-2">
                    <FileText size={16} className="text-muted-foreground" />
                    <span className="text-sm text-muted-foreground">
                        Requirements
                    </span>
                </div>
                <Button
                    onClick={handleGenerateTests}
                    disabled={isGenerating}
                    className="group transition-all"
                >
                    {isGenerating ? (
                        <div className="flex items-center gap-2">
                            <div className="h-4 w-4 rounded-full border-2 border-current border-r-transparent animate-spin" />
                            <span>Generating...</span>
                        </div>
                    ) : (
                        <div className="flex items-center gap-2">
                            <CheckSquare size={16} />
                            <span>Generate Tests</span>
                            <PenLine
                                size={16}
                                className="ml-auto transition-transform group-hover:translate-x-1"
                            />
                        </div>
                    )}
                </Button>
            </CardFooter>
        </Card>
    );
};
