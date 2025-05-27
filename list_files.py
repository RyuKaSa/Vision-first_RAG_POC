import subprocess

def list_unignored_files():
    try:
        result = subprocess.run(
            ['git', 'ls-files', '--others', '--exclude-standard', '--cached'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        files = result.stdout.strip().split('\n')
        return [f for f in files if f]  # filter out empty strings
    except subprocess.CalledProcessError as e:
        print("Error:", e.stderr)
        return []

if __name__ == "__main__":
    files = list_unignored_files()
    for f in files:
        print(f)
