import json
from typing import List

from issue import Issue

versions = ["2020.2", "2020.3", "2021.1", "2021.2", "2021.3"]


class IssueLoader:
    def __init__(self, issues: List[Issue]):
        self._issues = issues

    def get_by_version(self, version: str) -> List[Issue]:
        return [issue for issue in self._issues if version == issue.version]

    def get_versions(self) -> List[str]:
        return list({issue.version for issue in self._issues})

    @property
    def issues(self) -> List[Issue]:
        return self._issues

    def __len__(self) -> int:
        return len(self._issues)

    @staticmethod
    def from_json(json_path: str) -> "IssueLoader":
        issues = []

        with open(json_path) as file:
            for line in file:
                issue_jdict = json.loads(line)
                for version in issue_jdict["Affected versions"]:
                    if version in versions:
                        issue = Issue(id=issue_jdict["idReadable"],
                                      ts=int(issue_jdict["created"]),
                                      summary=issue_jdict["summary"],
                                      description=issue_jdict["description"],
                                      version=version)
                        issues.append(issue)

        return IssueLoader(issues)
