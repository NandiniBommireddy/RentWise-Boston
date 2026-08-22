"""Provenance registry for every dataset RentWise ingests.

Citations are only worth anything if a judge (or a resident) can click through to the
row we used. Every fact the backend emits carries a SourceRef built from this table,
so the path from an answer sentence back to data.boston.gov is never guessed.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Dataset:
    key: str
    title: str
    package: str
    resource_id: str
    csv: str
    table: str
    update_frequency: str
    license: str

    @property
    def landing_url(self) -> str:
        return f"https://data.boston.gov/dataset/{self.package}"

    @property
    def resource_url(self) -> str:
        return f"{self.landing_url}/resource/{self.resource_id}"

    def record_url(self, record_id: int | str) -> str:
        """Deep link to a single row via the CKAN datastore SQL API."""
        return (
            "https://data.boston.gov/api/3/action/datastore_search"
            f"?resource_id={self.resource_id}&filters=%7B%22_id%22%3A{record_id}%7D"
        )


RENTSMART = Dataset(
    key="rentsmart",
    title="RentSmart Boston (2016–present)",
    package="rentsmart",
    resource_id="dc615ff7-2ff3-416a-922b-f0f334f085d0",
    csv="rentsmart_2016_present.csv",
    table="rentsmart",
    update_frequency="daily",
    license="Open Data Commons PDDL",
)

STR_ELIGIBILITY = Dataset(
    key="str_eligibility",
    title="Short-Term Rental Eligibility",
    package="short-term-rental-eligibility",
    resource_id="83621b97-9a00-4aa7-bf43-28cae04969d4",
    csv="str_eligibility.csv",
    table="str_eligibility",
    update_frequency="daily (refreshed nightly)",
    license="Open Data Commons PDDL",
)

INCOME_RESTRICTED = Dataset(
    key="income_restricted",
    title="Income-Restricted Housing Inventory 2022",
    package="income-restricted-housing",
    resource_id="bfdeffc7-ab5e-4e28-9561-d4e84c9674e4",
    csv="income_restricted_2022.csv",
    table="income_restricted",
    update_frequency="annual",
    license="Open Data Commons PDDL",
)

DATASETS = {d.key: d for d in (RENTSMART, STR_ELIGIBILITY, INCOME_RESTRICTED)}

# Datasets required to be live for the app to serve. Income-restricted is ingested
# opportunistically -- it is a 1.5k-row lookup table, not a primary retrieval target.
REQUIRED = (RENTSMART, STR_ELIGIBILITY)


@dataclass(frozen=True)
class SourceRef:
    """One citation: which dataset, which row, and what it said."""

    dataset_key: str
    record_id: int | str | None
    label: str
    detail: str = ""

    @property
    def dataset(self) -> Dataset:
        return DATASETS[self.dataset_key]

    def to_dict(self) -> dict:
        ds = self.dataset
        return {
            "dataset": ds.title,
            "dataset_key": ds.key,
            "record_id": self.record_id,
            "label": self.label,
            "detail": self.detail,
            "url": ds.record_url(self.record_id) if self.record_id is not None else ds.resource_url,
            "updated": ds.update_frequency,
            "license": ds.license,
        }

    def to_line(self) -> str:
        """Compact single-line form, for the frontend's current sources?: string[]."""
        ds = self.dataset
        rid = f" #{self.record_id}" if self.record_id is not None else ""
        detail = f" — {self.detail}" if self.detail else ""
        return f"{ds.title}{rid}: {self.label}{detail}"
