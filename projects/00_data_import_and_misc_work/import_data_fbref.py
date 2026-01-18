"""Script to download and save fbref data using ScraperFC 4.x API.

This script scrapes football statistics from FBref using the ScraperFC package.
It extracts player and team IDs from HTML for proper deduplication and merging.

Usage:
    1. Set COMPETITION to one of the valid league names (see VALID_LEAGUES below)
    2. Set COMPETITION_END_YEAR to the year the season ends (e.g., 2024 for 2023-2024)
    3. Set STORAGE_MODE to control what data is saved
    4. Run the script

Valid league names (ScraperFC 4.x):
    - Argentina Liga Profesional
    - Australia A-League Women
    - Belgium Pro League
    - Brazil Serie A
    - CONMEBOL Copa America
    - CONMEBOL Copa Libertadores
    - England EFL Championship
    - England Premier League
    - England WSL
    - FBref Big 5 Combined
    - FIFA Womens World Cup
    - FIFA World Cup
    - France Ligue 1
    - France Ligue 2
    - France Premiere Ligue
    - Germany 2.Bundesliga
    - Germany Bundesliga
    - Germany Womens Bundesliga
    - Italy Serie A
    - Italy Serie B
    - Italy Womens Serie A
    - Mexico Liga MX
    - Netherlands Eredivisie
    - Portugal Primeira Liga
    - Saudi Arabia Pro League
    - Spain La Liga
    - Spain La Liga 2
    - Spain Liga F
    - Turkiye Super Lig
    - UEFA Champions League
    - UEFA Conference League
    - UEFA Europa League
    - UEFA European Championship
    - UEFA Womens Champions League
    - UEFA Womens European Championship
    - USA MLS
    - USA NWSL
    - USA NWSL Challenge Cup
    - USA NWSL Fall Series
"""

# %% Imports

import re
from io import StringIO
from pathlib import Path

import pandas as pd
import ScraperFC as sfc
from bs4 import BeautifulSoup
from ScraperFC.fbref import stats_categories, comps

# %% User inputs

# Select competition - use full ScraperFC 4.x league name (see docstring for valid names)
COMPETITION = "England Premier League"

# Select calendar year in which the competition finishes (e.g., 2024 for 2023-2024 season)
COMPETITION_END_YEAR = 2023

# Select whether to store player data, team data or vs team data
# Options: 'player_only', 'team_only', 'vs_team_only', 'all'
STORAGE_MODE = "all"

# %% Helper functions


def get_script_dir() -> Path:
    """Get the directory containing this script or current working directory."""
    try:
        # Works when running as a script
        return Path(__file__).resolve().parent
    except NameError:
        # Fallback for interactive/REPL environments
        return Path.cwd()


def get_output_dir(competition_end_year: int, competition: str) -> Path:
    """Create and return the output directory path using the competition name."""
    season_str = f"{competition_end_year - 1}_{str(competition_end_year)[2:]}"

    # Try script-relative path first
    try:
        script_dir = Path(__file__).resolve().parent
        base_dir = script_dir / ".." / ".." / "data_directory"
    except NameError:
        # Fallback for interactive/REPL: search for the external project structure
        cwd = Path.cwd()
        external_data_dir = (
            cwd / "external" / "jakeyk11-football-data-analytics" / "data_directory"
        )
        if (
            external_data_dir.exists()
            or (cwd / "external" / "jakeyk11-football-data-analytics").exists()
        ):
            base_dir = external_data_dir
        else:
            # Last resort: use cwd/data_directory
            base_dir = cwd / "data_directory"

    output_dir = (base_dir / "fbref_data" / season_str / competition).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def validate_league(competition: str) -> None:
    """Validate that the competition name is valid for ScraperFC 4.x."""
    if competition not in comps:
        valid_leagues = sorted(comps.keys())
        raise ValueError(
            f"Invalid league: '{competition}'\n"
            f"Valid leagues are:\n  " + "\n  ".join(valid_leagues)
        )


def get_season_string(end_year: int) -> str:
    """Convert end year to season string format (e.g., 2024 -> '2023-2024')."""
    return f"{end_year - 1}-{end_year}"


def flatten_multiindex_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten MultiIndex columns to single-level column names."""
    if not isinstance(df.columns, pd.MultiIndex):
        return df

    new_col_names = []
    for col in df.columns:
        if isinstance(col, tuple):
            col_name_1 = "" if "Unnamed" in str(col[0]) else str(col[0])
            col_name_2 = str(col[1]) if "Unnamed" in str(col[0]) else " " + str(col[1])
            new_col_names.append((col_name_1 + col_name_2).strip())
        else:
            new_col_names.append(str(col))
    df.columns = new_col_names

    # Remove header rows that get parsed as data rows (where Rk == "Rk")
    if "Rk" in df.columns:
        df = df[df["Rk"] != "Rk"].reset_index(drop=True)

    return df


def extract_ids_from_soup(soup_table: BeautifulSoup, id_type: str) -> dict:
    """
    Extract player or team IDs from BeautifulSoup table.

    Args:
        soup_table: BeautifulSoup table element
        id_type: 'player' or 'squad' (team)

    Returns:
        Dict mapping names to IDs
    """
    id_map = {}
    if soup_table is None:
        return id_map

    data_stat = id_type if id_type == "player" else "team"
    pattern = rf"/en/({'players' if id_type == 'player' else 'squads'})/([a-f0-9]+)/"

    # For players, check td elements
    if id_type == "player":
        for td in soup_table.find_all("td", {"data-stat": data_stat}):
            link = td.find("a")
            if link and link.get("href"):
                match = re.search(pattern, link["href"])
                if match:
                    name = link.text.strip()
                    entity_id = match.group(2)
                    id_map[name] = entity_id
    else:
        # For teams/squads, check th elements (team names are in header cells)
        for th in soup_table.find_all("th", {"data-stat": data_stat}):
            link = th.find("a")
            if link and link.get("href"):
                match = re.search(pattern, link["href"])
                if match:
                    name = link.text.strip()
                    entity_id = match.group(2)
                    id_map[name] = entity_id

    return id_map


def add_ids_to_dataframe(
    df: pd.DataFrame, id_map: dict, name_col: str, id_col: str
) -> pd.DataFrame:
    """Add ID column to DataFrame based on name mapping."""
    if not id_map or name_col not in df.columns:
        return df
    df[id_col] = df[name_col].map(id_map)
    return df


def extract_player_team_ids_from_soup(soup_table: BeautifulSoup) -> list:
    """
    Extract team IDs for each player row from BeautifulSoup table.

    Returns list of (player_id, team_id) tuples in row order.
    This handles players who played for multiple teams correctly.
    """
    rows_data = []
    if soup_table is None:
        return rows_data

    team_pattern = r"/en/squads/([a-f0-9]+)/"
    player_pattern = r"/en/players/([a-f0-9]+)/"

    # Find all data rows in tbody (skip header rows)
    tbody = soup_table.find("tbody")
    if not tbody:
        return rows_data

    for row in tbody.find_all("tr"):
        # Skip rows that are section headers (have class="thead" or similar)
        if row.get("class") and any("thead" in c for c in row.get("class", [])):
            continue

        player_cell = row.find(["td", "th"], {"data-stat": "player"})
        team_cell = row.find(["td", "th"], {"data-stat": "team"})

        player_id = None
        team_id = None

        if player_cell:
            player_link = player_cell.find("a")
            if player_link:
                player_href = player_link.get("href", "")
                player_match = re.search(player_pattern, player_href)
                if player_match:
                    player_id = player_match.group(1)

        if team_cell:
            team_link = team_cell.find("a")
            if team_link:
                team_href = team_link.get("href", "")
                team_match = re.search(team_pattern, team_href)
                if team_match:
                    team_id = team_match.group(1)

        rows_data.append((player_id, team_id))

    return rows_data


def add_team_id_to_player_df(df: pd.DataFrame, rows_data: list) -> pd.DataFrame:
    """Add Team ID column to player DataFrame using row-level data."""
    if not rows_data:
        return df

    # The rows_data list should match the DataFrame row order
    # Filter out any header rows that might have been included
    valid_rows = [r for r in rows_data if r[0] is not None]

    if len(valid_rows) != len(df):
        # Fallback: try to map by player ID
        id_map = {pid: tid for pid, tid in valid_rows if pid and tid}
        if "Player ID" in df.columns:
            # For each row, find matching team_id from the first occurrence
            df["Team ID"] = df["Player ID"].map(id_map)
        return df

    # Direct mapping by position
    df["Team ID"] = [r[1] for r in valid_rows]
    return df


def get_merge_keys(df: pd.DataFrame, data_type: str) -> list:
    """Determine the best merge keys based on available columns."""
    if data_type == "team":
        # Prefer Team ID as it's consistent across stat categories
        if "Team ID" in df.columns:
            return ["Team ID"]
        elif "Squad" in df.columns:
            return ["Squad"]
        else:
            return []

    elif data_type == "player":
        # Use Player ID + Team ID for uniqueness
        # This handles players who transferred mid-season AND inconsistent squad names
        keys = []
        if "Player ID" in df.columns:
            keys.append("Player ID")
        if "Team ID" in df.columns:
            keys.append("Team ID")
        # Fallback to Player name + Squad if no IDs
        if not keys:
            if "Player" in df.columns:
                keys.append("Player")
            if "Squad" in df.columns:
                keys.append("Squad")
        return keys

    return []


def safe_merge(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    merge_keys: list,
    suffixes: tuple = ("", "_duplicate"),
) -> pd.DataFrame:
    """
    Safely merge DataFrames avoiding cartesian products.

    Drops duplicates on merge keys before merging and validates result.
    """
    if left_df.empty:
        return right_df
    if right_df.empty:
        return left_df
    if not merge_keys:
        return left_df

    # Filter to only existing keys
    existing_keys = [
        k for k in merge_keys if k in left_df.columns and k in right_df.columns
    ]
    if not existing_keys:
        return left_df

    # Drop duplicates on merge keys to prevent cartesian product
    right_deduped = right_df.drop_duplicates(subset=existing_keys)

    # Perform merge
    merged = left_df.merge(
        right_deduped,
        on=existing_keys,
        suffixes=suffixes,
        how="outer",
    )

    # Warn if significant row increase (potential cartesian product)
    if len(merged) > len(left_df) * 1.5 and len(merged) > 100:
        print(f"Warning: Merge increased rows from {len(left_df)} to {len(merged)}")

    return merged


def remove_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove columns with '_duplicate' suffix."""
    return df.loc[:, [col for col in df.columns if "_duplicate" not in col]]


def convert_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Try to convert columns to numeric where possible."""
    for col in df.columns:
        try:
            # Try conversion and only apply if it succeeds
            converted = pd.to_numeric(df[col])
            df[col] = converted
        except (ValueError, TypeError):
            pass
    return df


class FBrefScraperWithIDs(sfc.FBref):
    """Extended FBref scraper that extracts player and team IDs."""

    def scrape_stats_with_ids(
        self, year: str, league: str, stat_category: str
    ) -> dict[str, pd.DataFrame]:
        """
        Scrape stats with player and team IDs extracted from HTML.

        Returns dict with 'squad', 'opponent', 'player' DataFrames.
        """
        valid_seasons = self.get_valid_seasons(league)
        if year not in valid_seasons:
            raise ValueError(f"Invalid year {year} for {league}")

        if stat_category not in stats_categories:
            raise ValueError(f"Invalid stat category: {stat_category}")

        season_url = valid_seasons[year]
        is_big5 = "big 5 combined" in league.lower()

        # Build URL
        season_url_split = season_url.split("/")
        season_url_split.insert(-1, stats_categories[stat_category]["url"])
        if is_big5:
            season_url_split.insert(-1, "squads")
        stat_url = "/".join(season_url_split)

        soup = self._get_soup(stat_url)

        # Extract squad stats
        squad_tag = soup.find(
            "table",
            {"id": re.compile(f"{stats_categories[stat_category]['html']}_for")},
        )
        squad_df = (
            pd.read_html(StringIO(str(squad_tag)))[0] if squad_tag else pd.DataFrame()
        )
        team_ids = extract_ids_from_soup(squad_tag, "squad")
        squad_df = flatten_multiindex_columns(squad_df)
        squad_df = add_ids_to_dataframe(squad_df, team_ids, "Squad", "Team ID")

        # Extract opponent stats
        opp_tag = soup.find(
            "table",
            {"id": re.compile(f"{stats_categories[stat_category]['html']}_against")},
        )
        opp_df = pd.read_html(StringIO(str(opp_tag)))[0] if opp_tag else pd.DataFrame()
        opp_team_ids = extract_ids_from_soup(opp_tag, "squad")
        opp_df = flatten_multiindex_columns(opp_df)
        opp_df = add_ids_to_dataframe(opp_df, opp_team_ids, "Squad", "Team ID")

        # For Big 5, need to fetch player stats from different URL
        if is_big5:
            player_url_split = season_url.split("/")
            player_url_split.insert(-1, stats_categories[stat_category]["url"])
            player_url_split.insert(-1, "players")
            player_url = "/".join(player_url_split)
            soup = self._get_soup(player_url)

        # Extract player stats
        player_tag = soup.find(
            "table", {"id": f"stats_{stats_categories[stat_category]['html']}"}
        )
        player_df = (
            pd.read_html(StringIO(str(player_tag)))[0] if player_tag else pd.DataFrame()
        )
        player_ids = extract_ids_from_soup(player_tag, "player")
        player_team_ids = extract_player_team_ids_from_soup(player_tag)
        player_df = flatten_multiindex_columns(player_df)
        player_df = add_ids_to_dataframe(player_df, player_ids, "Player", "Player ID")
        player_df = add_team_id_to_player_df(player_df, player_team_ids)

        return {"squad": squad_df, "opponent": opp_df, "player": player_df}

    def scrape_all_stats_with_ids(self, year: str, league: str) -> dict:
        """Scrape all stat categories with IDs."""
        import time
        from tqdm import tqdm

        return_package = {}
        for stat_category in tqdm(stats_categories, desc=f"{year} {league} stats"):
            start_time = time.time()
            stats = self.scrape_stats_with_ids(year, league, stat_category)
            elapsed = time.time() - start_time
            if elapsed < self.wait_time:
                time.sleep(self.wait_time - elapsed)
            return_package[stat_category] = stats

        return return_package


# %% Scrape data

if __name__ == "__main__":
    # Validate league name
    validate_league(COMPETITION)

    season_str = get_season_string(COMPETITION_END_YEAR)

    print(f"Scraping {COMPETITION} season {season_str}...")

    # Initialize scraper with ID extraction
    scraper = FBrefScraperWithIDs()

    # Get data
    fbref_dict = scraper.scrape_all_stats_with_ids(year=season_str, league=COMPETITION)

    # %% Format scraped data

    playerinfo_df = pd.DataFrame()
    teaminfo_for_df = pd.DataFrame()
    teaminfo_against_df = pd.DataFrame()

    # Iterate over statistic types
    for idx, statistic_group in enumerate(fbref_dict.keys()):
        stat_data = fbref_dict[statistic_group]

        # Team stats for
        temp_team_for = stat_data["squad"].copy()
        if not temp_team_for.empty:
            merge_keys = get_merge_keys(temp_team_for, "team")
            teaminfo_for_df = safe_merge(teaminfo_for_df, temp_team_for, merge_keys)

        # Team stats against
        temp_team_against = stat_data["opponent"].copy()
        if not temp_team_against.empty:
            merge_keys = get_merge_keys(temp_team_against, "team")
            teaminfo_against_df = safe_merge(
                teaminfo_against_df, temp_team_against, merge_keys
            )

        # Player stats
        temp_player = stat_data["player"].copy()
        if not temp_player.empty:
            merge_keys = get_merge_keys(temp_player, "player")
            playerinfo_df = safe_merge(playerinfo_df, temp_player, merge_keys)

    # Remove duplicate columns
    teaminfo_for_df = remove_duplicate_columns(teaminfo_for_df)
    teaminfo_against_df = remove_duplicate_columns(teaminfo_against_df)
    playerinfo_df = remove_duplicate_columns(playerinfo_df)

    # Remove incomplete player rows (rows without Player name are artifacts from outer merge)
    if "Player" in playerinfo_df.columns:
        incomplete_count = playerinfo_df["Player"].isna().sum()
        if incomplete_count > 0:
            print(
                f"Removing {incomplete_count} incomplete player rows (no Player name)"
            )
            playerinfo_df = playerinfo_df[playerinfo_df["Player"].notna()].reset_index(
                drop=True
            )

    # Convert numeric columns
    playerinfo_df = convert_numeric_columns(playerinfo_df)
    teaminfo_for_df = convert_numeric_columns(teaminfo_for_df)
    teaminfo_against_df = convert_numeric_columns(teaminfo_against_df)

    # %% Save scraped data

    output_dir = get_output_dir(COMPETITION_END_YEAR, COMPETITION)
    file_prefix = f"{COMPETITION.lower()} {COMPETITION_END_YEAR}"

    print(f"Saving data to {output_dir}...")
    print(f"  Player data: {len(playerinfo_df)} rows")
    print(f"  Team data: {len(teaminfo_for_df)} rows")
    print(f"  Vs Team data: {len(teaminfo_against_df)} rows")

    storage_mode = STORAGE_MODE.lower().replace("_", " ")

    if storage_mode in ("player only", "all"):
        playerinfo_df.to_json(output_dir / f"{file_prefix} player data.json")

    if storage_mode in ("team only", "all"):
        teaminfo_for_df.to_json(output_dir / f"{file_prefix} team data.json")

    if storage_mode in ("vs team only", "all"):
        teaminfo_against_df.to_json(output_dir / f"{file_prefix} vs team data.json")

    print("Done!")
