/* eslint-disable @typescript-eslint/no-explicit-any */
import { getInstance } from "@/api/instance";
import { useMutation, useQuery } from "@tanstack/react-query";

export function useJiraAuth(
    onSuccess?: (data: any) => any,
    onError?: () => any
) {
    return useMutation({
        mutationKey: ["jiraAuth"],
        mutationFn: async ({
            email,
            token,
            domain,
        }: {
            email: string;
            token: string;
            domain: string;
        }) => {
            try {
                const res = await getInstance().post("/authenticate", {
                    domain: domain,
                    email: email,
                    jira_token: token,
                });
                return res.data;
            } catch (error) {
                return {
                    message:
                        "Error authenticating with JIRA, Please check your credentials",
                };
            }
        },
        onSuccess: onSuccess,
        onError: onError,
    });
}

export function useStories(
    onSuccess?: (data: any) => any,
    onError?: () => any
) {
    return useMutation({
        mutationKey: ["stories"],
        mutationFn: async ({
            id,
            email,
            token,
            domain,
        }: {
            id: string;
            email: string;
            token: string;
            domain: string;
        }) => {
            try {
                const res = await getInstance().post("/fetch-stories", {
                    domain: domain,
                    email: email,
                    jira_id: id,
                    jira_token: token,
                });
                return res.data;
            } catch (error) {
                return {
                    message:
                        "Error connecting with JIRA, Please check your credentials for JIRA ID",
                };
            }
        },
        onSuccess: onSuccess,
        onError: onError,
    });
}

export function useGenerateTestCases(
    onSuccess?: (data: any) => any,
    onError?: () => any
) {
    return useMutation({
        mutationKey: ["generateTestCases"],
        mutationFn: async ({
            jiraId,
            userStory,
            acceptanceCriteria,
        }: {
            jiraId: string;
            userStory: string;
            acceptanceCriteria?: string;
        }) => {
            try {
                const res = await getInstance().post("/generate-test-cases", {
                    jira_id: jiraId,
                    user_story: userStory,
                    acceptance_criteria: acceptanceCriteria,
                });
                return res.data;
            } catch (error) {
                return {
                    message: "Error generating test cases",
                };
            }
        },
        onSuccess: onSuccess,
        onError: onError,
    });
}
