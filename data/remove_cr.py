import os

def remove_cr(ifname, ofname):
    with open(ifname, "rb") as f_in:   # バイナリで読む
        data = f_in.read()

    # CRLF を LF に置き換える
    data = data.replace(b"\r", b"")

    with open(ofname, "wb") as f_out:
        f_out.write(data)

if __name__ == "__main__":
    output_dir = "fixed_audiocaps"
    os.makedirs(output_dir, exist_ok=True)
    for fname in ["train.csv", "val.csv", "test.csv"]:
        remove_cr(
            os.path.join("audiocaps/dataset", fname),
            os.path.join(output_dir, fname)
        )
