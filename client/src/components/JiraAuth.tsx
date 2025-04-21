import { useState } from "react";
import {
    Card,
    CardContent,
    CardDescription,
    CardFooter,
    CardHeader,
    CardTitle,
} from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { ArrowRight, Lock } from "lucide-react";
import { toast } from "sonner";
import { useJiraAuth, useStories } from "@/hooks/jira";

interface JiraAuthProps {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    onAuthenticate: (stories: any[]) => void;
}

export const JiraAuth = ({ onAuthenticate }: JiraAuthProps) => {
    const [domain, setDomain] = useState("");
    const [email, setEmail] = useState("");
    const [jiraId, setJiraID] = useState("");
    const [token, setToken] = useState("");
    const [isAuthenticating, setIsAuthenticating] = useState(false);
    const [fetchingStories, setFetchingStories] = useState(false);

    // Auth mutation
    const { mutate: authenticateJira } = useJiraAuth(
        () => {
            setIsAuthenticating(false);
            toast.success("Successfully authenticated with JIRA");
            // After authentication success, fetch stories
            fetchStories();
        },
        () => {
            setIsAuthenticating(false);
            toast.error(
                "Failed to authenticate with JIRA. Please check your credentials."
            );
        }
    );

    // Stories mutation
    const { mutate: fetchStoriesMutation } = useStories(
        (data) => {
            setFetchingStories(false);
            toast.success("Successfully fetched JIRA stories");

            // Transform stories from the API response to the format expected by the component
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            const transformedStories = data.stories.map((story: any) => ({
                id: story.key,
                title: story.summary,
                description: story.description || "No description provided",
                priority: story.priority || "Medium",
                status: story.status || "To Do",
                assignee: story.assignee || "Unassigned",
                dueDate: story.due_date || "No due date",
                epicLink: story.epic_link,
                tags: story.tags || [],
                onGenerateTests: () => {}, // Will be overwritten by the parent component
            }));

            onAuthenticate(transformedStories);
        },
        () => {
            setFetchingStories(false);
            toast.error("Failed to fetch stories from JIRA.");
        }
    );

    const fetchStories = () => {
        setFetchingStories(true);
        fetchStoriesMutation({
            domain: domain,
            email: email,
            id: jiraId,
            token: token,
        });
    };

    const handleSubmit = (e: React.FormEvent) => {
        e.preventDefault();

        if (!domain || !email || !token || !jiraId) {
            toast.error("Please fill in all fields");
            return;
        }

        setIsAuthenticating(true);
        authenticateJira({
            domain: domain,
            email: email,
            token: token,
        });
    };

    const isPending = isAuthenticating || fetchingStories;
    const buttonText = isAuthenticating
        ? "Authenticating..."
        : fetchingStories
        ? "Fetching Stories..."
        : "Connect Securely";

    return (
        <Card className="w-full max-w-md mx-auto glass-card animate-scale-in">
            <CardHeader className="space-y-1">
                <CardTitle className="text-2xl font-medium">
                    Connect to JIRA
                </CardTitle>
                <CardDescription>
                    Enter your JIRA credentials and JIRA EPIC ID to fetch user
                    stories
                </CardDescription>
            </CardHeader>
            <CardContent>
                <form onSubmit={handleSubmit} className="space-y-4">
                    <div className="space-y-2">
                        <Label htmlFor="domain">JIRA Domain</Label>
                        <Input
                            id="domain"
                            placeholder="your-company.atlassian.net"
                            value={domain}
                            onChange={(e) => setDomain(e.target.value)}
                            className="transition-all duration-200 focus:ring-2 focus:ring-primary/30"
                        />
                    </div>
                    <div className="space-y-2">
                        <Label htmlFor="email">Email</Label>
                        <Input
                            id="email"
                            type="email"
                            placeholder="your-email@example.com"
                            value={email}
                            onChange={(e) => setEmail(e.target.value)}
                            className="transition-all duration-200 focus:ring-2 focus:ring-primary/30"
                        />
                    </div>
                    <div className="space-y-2">
                        <Label htmlFor="epicId">EPIC ID</Label>
                        <Input
                            id="epicId"
                            type="text"
                            placeholder="PROJ-123"
                            value={jiraId}
                            onChange={(e) => setJiraID(e.target.value)}
                            className="transition-all duration-200 focus:ring-2 focus:ring-primary/30"
                        />
                    </div>
                    <div className="space-y-2">
                        <div className="flex items-center justify-between">
                            <Label htmlFor="token">API Token</Label>
                            <a
                                href="https://id.atlassian.com/manage/api-tokens"
                                target="_blank"
                                rel="noreferrer"
                                className="text-xs text-primary hover:underline highlight-link"
                            >
                                Create a token
                            </a>
                        </div>
                        <Input
                            id="token"
                            type="password"
                            placeholder="••••••••••••••••"
                            value={token}
                            onChange={(e) => setToken(e.target.value)}
                            className="transition-all duration-200 focus:ring-2 focus:ring-primary/30"
                        />
                    </div>

                    <Button
                        type="submit"
                        className="w-full group"
                        disabled={isPending}
                    >
                        {isPending ? (
                            <div className="flex items-center gap-2">
                                <div className="h-4 w-4 rounded-full border-2 border-current border-r-transparent animate-spin" />
                                <span>{buttonText}</span>
                            </div>
                        ) : (
                            <div className="flex items-center gap-2">
                                <Lock size={16} />
                                <span>Connect Securely</span>
                                <ArrowRight
                                    size={16}
                                    className="ml-auto transition-transform group-hover:translate-x-1"
                                />
                            </div>
                        )}
                    </Button>
                </form>
            </CardContent>
            <CardFooter className="px-6 py-4 bg-secondary/30 rounded-b-lg border-t">
                <p className="text-xs text-muted-foreground text-center w-full">
                    Your API token is stored locally and never sent to our
                    servers. We connect directly to JIRA's API.
                </p>
            </CardFooter>
        </Card>
    );
};
