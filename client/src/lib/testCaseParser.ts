/**
 * Parse AI generated test cases from markdown format into structured objects
 */
export interface ParsedTestCase {
    id: string;
    title: string;
    steps: string[];
    expectedResult: string;
    preconditions?: string;
    testData?: string;
    priority?: string;
    references?: string;
    scenario?: string;
    scenarioId?: string;
    passCriteria?: string;
    failCriteria?: string;
}

export function parseAIGeneratedTestCases(content: string): ParsedTestCase[] {
    const testCases: ParsedTestCase[] = [];

    // Extract test scenarios to associate with test cases
    const scenarioMap = new Map<string, { id: string; title: string }>();
    const scenarioRegex =
        /#### \*\*Test Scenario ID: (TS_\d+)\*\*\s*\n\*\*Test Scenario:\*\* (.*?)(?=\n)/g;
    let scenarioMatch;
    while ((scenarioMatch = scenarioRegex.exec(content)) !== null) {
        scenarioMap.set(scenarioMatch[1], {
            id: scenarioMatch[1],
            title: scenarioMatch[2].trim(),
        });
    }

    // Extract test cases with detailed format
    const testCaseRegex =
        /##### \*\*Test Case ID: (TC_\d+)\*\*\s*\n- \*\*Test Case:\*\* (.*?)\n(?:- \*\*Preconditions:\*\* (.*?)\n)?(?:- \*\*Test Data:\*\* (.*?)\n)?- \*\*Test Execution Steps:\*\*\s*\n([\s\S]*?)(?=- \*\*Expected Outcome:|\*\*Expected Outcome:)(?:- )?\*\*Expected Outcome:\*\*\s*([\s\S]*?)(?=- \*\*Pass\/Fail Criteria:|-\s\*\*Pass\/Fail|\*\*Pass\/Fail)(?:- )?\*\*Pass\/Fail Criteria:\*\*\s*\n(?:\s*- \*\*Pass:\*\* (.*?)\n)?(?:\s*- \*\*Fail:\*\* (.*?)\n)?(?:- \*\*Priority:\*\* (.*?)\n)?(?:- \*\*References:\*\* (.*?)(?=\n\n|\n#|\n$|$))?/g;

    let match;
    let currentScenarioId = "";

    // Find the current scenario ID for each test case
    const findScenarioForTestCase = (position: number): string => {
        const contentBeforeTestCase = content.substring(0, position);
        const scenarioMatches = [
            ...contentBeforeTestCase.matchAll(
                /#### \*\*Test Scenario ID: (TS_\d+)\*\*/g
            ),
        ];
        if (scenarioMatches.length > 0) {
            return scenarioMatches[scenarioMatches.length - 1][1];
        }
        return "";
    };

    while ((match = testCaseRegex.exec(content)) !== null) {
        currentScenarioId = findScenarioForTestCase(match.index);

        const id = match[1].trim();
        const title = match[2].trim();
        const preconditions = match[3]?.trim();
        const testData = match[4]?.trim();

        // Parse steps - each step is numbered with a period
        const stepsText = match[5]?.trim() || "";
        const steps: string[] = [];

        // Extract steps using regex for numbered list items
        const stepRegex = /\d+\.\s*(.*?)(?=\n\s*\d+\.|\n\s*$|$)/gs;
        let stepMatch;
        while ((stepMatch = stepRegex.exec(stepsText)) !== null) {
            steps.push(stepMatch[1].trim());
        }

        const expectedResult = match[6]?.trim() || "";
        const passCriteria = match[7]?.trim();
        const failCriteria = match[8]?.trim();
        const priority = match[9]?.trim();
        const references = match[10]?.trim();

        const scenario = scenarioMap.get(currentScenarioId);

        testCases.push({
            id,
            title,
            steps,
            expectedResult,
            preconditions,
            testData,
            priority,
            references,
            scenarioId: currentScenarioId,
            scenario: scenario?.title || "",
            passCriteria,
            failCriteria,
        });
    }

    return testCases;
}
