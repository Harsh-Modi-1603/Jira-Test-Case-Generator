import { useState } from "react";
import { JiraAuth } from "@/components/JiraAuth";
import { UserStory, UserStoryProps } from "@/components/UserStory";
import { TestCasesList } from "@/components/TestCasesList";
import { Separator } from "@/components/ui/separator";
import { SearchX, CircleDashed, RefreshCw } from "lucide-react";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import {
    parseAIGeneratedTestCases,
    ParsedTestCase,
} from "../lib/testCaseParser";

interface TestCase {
    id: string;
    title: string;
    steps: string[];
    expectedResult: string;
}

const Index = () => {
    const [isAuthenticated, setIsAuthenticated] = useState(false);
    const [isLoading, setIsLoading] = useState(false);
    const [searchTerm, setSearchTerm] = useState("");
    const [userStories, setUserStories] = useState<UserStoryProps[]>([]);
    const [selectedStoryId, setSelectedStoryId] = useState<string | null>(null);
    const [testCases, setTestCases] = useState<{ [key: string]: TestCase[] }>(
        {}
    );
    const [rawResponses, setRawResponses] = useState<{ [key: string]: string }>(
        {}
    );

    // Credentials cache for regenerating test cases or re-authentication
    const [credentialsCache, setCredentialsCache] = useState({
        domain: "",
        email: "",
        token: "",
        epicId: "",
    });

    const handleAuthenticate = (stories: UserStoryProps[]) => {
        // Map stories to include the handleGenerateTests function
        const storiesWithHandlers = stories.map((story) => ({
            ...story,
            onGenerateTests: handleGenerateTestCasesSubmission,
        }));

        setUserStories(storiesWithHandlers);
        setIsAuthenticated(true);
        setIsLoading(false);
    };

    const handleGenerateTestCasesSubmission = async (
        id: string,
        rawResponse: { content: string; token_count: number }
    ) => {
        try {
            // Parse the AI response into structured test cases
            const parsedTestCases = parseAIGeneratedTestCases(
                rawResponse.content
            );

            // Store the test cases for this story
            setTestCases((prev) => ({
                ...prev,
                [id]: parsedTestCases,
            }));

            // Store the raw response content
            setRawResponses((prev) => ({
                ...prev,
                [id]: rawResponse.content,
            }));

            // Navigate to the test cases view
            setSelectedStoryId(id);
        } catch (error) {
            console.error("Failed to parse test cases:", error);
        }
    };

    const handleCloseTestCases = () => {
        setSelectedStoryId(null);
    };

    const handleSaveCredentials = (
        domain: string,
        email: string,
        token: string,
        epicId: string
    ) => {
        setCredentialsCache({
            domain,
            email,
            token,
            epicId,
        });
    };

    const filteredStories = userStories.filter(
        (story) =>
            story.id.toLowerCase().includes(searchTerm.toLowerCase()) ||
            story.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
            story.tags.some((tag) =>
                tag.toLowerCase().includes(searchTerm.toLowerCase())
            )
    );

    if (isLoading) {
        return (
            <div className="min-h-screen flex flex-col items-center justify-center p-6">
                <div className="flex flex-col items-center space-y-6 max-w-lg text-center">
                    <CircleDashed
                        size={40}
                        className="text-primary animate-spin"
                    />
                    <h2 className="text-2xl font-medium">
                        Loading JIRA Stories
                    </h2>
                    <p className="text-muted-foreground">
                        We're retrieving your stories from JIRA. This may take a
                        moment...
                    </p>
                </div>
            </div>
        );
    }

    if (!isAuthenticated) {
        return (
            <div className="min-h-screen flex flex-col items-center justify-center p-6 bg-gradient-to-b from-background to-secondary/50">
                <div className="w-full max-w-4xl space-y-10">
                    <JiraAuth onAuthenticate={handleAuthenticate} />

                    <div className="flex justify-center mt-8 text-center">
                        <p className="text-sm text-muted-foreground max-w-md">
                            Testcase Generator uses AI to analyze your JIRA
                            stories and generate detailed test cases across
                            functional, UI, performance, and security domains.
                        </p>
                    </div>
                </div>
            </div>
        );
    }

    if (selectedStoryId) {
        const currentTestCases = testCases[selectedStoryId] || [];
        const currentRawContent = rawResponses[selectedStoryId] || "";

        return (
            <div className="min-h-screen flex items-center justify-center p-6 bg-gradient-to-b from-background to-secondary/50">
                <TestCasesList
                    storyId={selectedStoryId}
                    testCases={currentTestCases}
                    rawContent={currentRawContent}
                    onClose={handleCloseTestCases}
                />
            </div>
        );
    }

    return (
        <div className="min-h-screen flex flex-col p-6 bg-gradient-to-b from-background to-secondary/50">
            <header className="w-full max-w-6xl mx-auto mb-8 animate-slide-down">
                <div className="flex flex-col space-y-4">
                    <div className="flex items-center justify-between">
                        <h1 className="text-3xl font-medium tracking-tight">
                            JIRA User Stories
                        </h1>
                        <Button
                            variant="outline"
                            size="sm"
                            onClick={() => setIsAuthenticated(false)}
                            className="gap-1.5"
                        >
                            <RefreshCw size={14} />
                            <span>Change JIRA Account</span>
                        </Button>
                    </div>

                    <div className="flex w-full max-w-sm items-center space-x-2">
                        <Input
                            placeholder="Search stories by ID, title or tag..."
                            value={searchTerm}
                            onChange={(e) => setSearchTerm(e.target.value)}
                            className="transition-all duration-200 focus:ring-2 focus:ring-primary/30"
                        />
                    </div>
                </div>
            </header>

            <Separator className="mb-8" />

            <main className="flex-1 w-full max-w-6xl mx-auto">
                {filteredStories.length > 0 ? (
                    <div className="grid gap-6 animate-fade-in">
                        {filteredStories.map((story) => (
                            <UserStory key={story.id} {...story} />
                        ))}
                    </div>
                ) : (
                    <div className="flex flex-col items-center justify-center h-96 text-center space-y-4">
                        <SearchX
                            size={48}
                            className="text-muted-foreground/50"
                        />
                        <h3 className="text-xl font-medium">
                            No stories found
                        </h3>
                        <p className="text-muted-foreground max-w-md">
                            No user stories match your search criteria. Try
                            adjusting your search term or refresh the connection
                            to JIRA.
                        </p>
                    </div>
                )}
            </main>
        </div>
    );
};

export default Index;
