# Use to retrieve data from mfe server, assumes a ssh key has already been setup with name mfe
import imaging_methods as im
import subprocess


def retrieve_data_from_tree(shot):
    print(f"Making data files for shot {shot}")

    try:
        subprocess.run(["ssh", "mfe", f"python3 gpi_apd.py {shot}"])
        result = subprocess.run(
            ["scp", "-pr", f"mfe:~/{shot}.nc", "data/"],
            check=True,  # Raise error on non-zero exit
            capture_output=True,  # Capture stdout/stderr
            text=True,  # Decode as text
        )
        print("Success! Stdout:", result.stdout)
    except subprocess.CalledProcessError as e:
        print("Error during SCP:", e.stderr)


if __name__ == "__main__":
    manager = im.GPIDataAccessor(
        "/home/sosno/Git/experimental_database/plasma_discharges.json"
    )

    retrieve_data_from_tree(1160616016)
