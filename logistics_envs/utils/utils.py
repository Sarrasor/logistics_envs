import datetime
import pandas as pd
import geopandas as gpd


def get_day_from_tlc_data(
    trip_record_path: str,
    zones_shapefile_path: str,
    target_day: str,
    sample_frac: float = 1.0,
) -> pd.DataFrame:
    """
    Extracts data for a specific day from TLC trip records.

    Args:
        trip_record_path (str): The path to a TLC trip record in Parquet format.
        zones_shapefile_path (str): The path to the shapefile containing zone information.
        target_day (str): The target day in the format "YYYY-MM-DD".

    Returns:
        pd.DataFrame: The extracted data for the target day.
    """

    df = pd.read_parquet(trip_record_path)
    zones_df = gpd.read_file(zones_shapefile_path)
    target_datetime = datetime.datetime.strptime(target_day, "%Y-%m-%d")
    next_datetime = target_datetime + datetime.timedelta(days=1)

    centroids = zones_df.geometry.centroid.to_crs(epsg=4326)
    zones_df["lat"] = centroids.y
    zones_df["lon"] = centroids.x
    zones_df = zones_df[["LocationID", "lat", "lon"]]

    df = df[["tpep_pickup_datetime", "PULocationID", "DOLocationID"]]
    df = df.rename(columns={"tpep_pickup_datetime": "creation_time"})
    df = df[(df["creation_time"] >= target_datetime) & (df["creation_time"] < next_datetime)]

    df = df.merge(zones_df, left_on="PULocationID", right_on="LocationID", how="left")
    df = df.rename(columns={"lat": "from_lat", "lon": "from_lon"})
    df = df.drop(columns=["LocationID", "PULocationID"])
    df = df.merge(zones_df, left_on="DOLocationID", right_on="LocationID", how="left")
    df = df.rename(columns={"lat": "to_lat", "lon": "to_lon"})
    df = df.drop(columns=["LocationID", "DOLocationID"])

    df = df.dropna()
    if sample_frac < 1.0:
        df = df.sample(frac=sample_frac)

    df["creation_time"] = (
        (df["creation_time"] - target_datetime).dt.total_seconds().div(60).astype(int)
    )
    df = df.sort_values(by="creation_time")
    df = df.reset_index(drop=True)

    return df
