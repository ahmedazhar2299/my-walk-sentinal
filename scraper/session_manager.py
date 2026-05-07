from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

from scraper.downloads import unique_path


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class StagedActivity:
    date: str
    session_index: int
    workout_type: str
    staged_path: Path
    original_filename: str
    recorded_time: str


class SessionManager:
    """Assign records to ordered sessions and materialize the required folder hierarchy."""

    def __init__(self, patient_dir: Path) -> None:
        self.patient_dir = patient_dir
        self._sessions_by_date: dict[str, list[dict[str, StagedActivity]]] = {}
        self._activities: list[StagedActivity] = []

    def stage(
        self,
        *,
        date: str,
        workout_type: str,
        staged_path: Path,
        original_filename: str,
        recorded_time: str,
    ) -> StagedActivity:
        sessions = self._sessions_by_date.setdefault(date, [])
        for index, session in enumerate(sessions, start=1):
            if workout_type not in session:
                activity = StagedActivity(date, index, workout_type, staged_path, original_filename, recorded_time)
                session[workout_type] = activity
                self._activities.append(activity)
                LOGGER.info("Assigned %s %s to activity_%s", date, workout_type, index)
                return activity

        session: dict[str, StagedActivity] = {}
        sessions.append(session)
        index = len(sessions)
        activity = StagedActivity(date, index, workout_type, staged_path, original_filename, recorded_time)
        session[workout_type] = activity
        self._activities.append(activity)
        LOGGER.info("Created activity_%s for %s and assigned %s", index, date, workout_type)
        return activity

    def materialize(self) -> None:
        for activity in self._activities:
            sessions_for_date = self._sessions_by_date[activity.date]
            if len(sessions_for_date) == 1:
                destination_dir = self.patient_dir / activity.date / activity.workout_type
            else:
                destination_dir = self.patient_dir / activity.date / f"activity_{activity.session_index}" / activity.workout_type

            destination = unique_path(destination_dir, activity.original_filename)
            if destination.exists():
                LOGGER.info("Skipping existing file: %s", destination)
                continue

            shutil.move(str(activity.staged_path), destination)
            LOGGER.info("Organized %s -> %s", activity.staged_path.name, destination)
