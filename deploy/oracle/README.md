# Run the BTC paper-trading bot on Oracle Cloud (free)

The bot needs about 180 MB of memory, almost no processor time, and a few MB of
internet a day. Oracle's "Always Free" server is more than enough, and it stays
on around the clock so the bot never misses a call.

Total time: about 30 minutes, most of it Oracle's signup.

## 1. Create the Oracle account

1. Go to https://www.oracle.com/cloud/free/ and click **Start for free**.
2. Fill in your details. Oracle asks for a card to confirm you are a real
   person. The Always Free servers are never charged.
3. When it asks for a **Home Region**, pick **Germany Central (Frankfurt)**.
   This choice is permanent. Do not pick a US region: Binance blocks its
   futures data for US servers, and the bot would get no prices.

## 2. Create the server

1. In the Oracle console, open the menu (top left) and go to
   **Compute > Instances > Create instance**.
2. Name it `paperbot`.
3. Under **Image and shape**, click **Edit**:
   - Image: **Canonical Ubuntu 24.04**.
   - Shape: **Ampere VM.Standard.A1.Flex** with 1 OCPU and 6 GB memory.
     If Oracle says it is out of capacity, use **VM.Standard.E2.1.Micro**
     instead. Both are free and both are big enough.
4. Under **Add SSH keys**, choose **Generate a key pair for me** and click
   **Save private key**. Keep that file. It is the only way into the server.
   Move it to `C:\Users\WorkHorizon\.ssh\paperbot.key`.
5. Click **Create**. After a minute or two the server shows **Running**.
   Copy its **Public IP address** (looks like `130.61.12.34`).

## 3. Lock the key file (Windows refuses keys anyone else can read)

Open PowerShell and run:

```powershell
icacls "$env:USERPROFILE\.ssh\paperbot.key" /inheritance:r /grant:r "$($env:USERNAME):R"
```

## 4. Stop the bot on your PC and pack it

The server will continue the same run, so the PC copy has to stop first or
both would trade. In PowerShell:

```powershell
Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -like '*forecast.paper*' } | ForEach-Object { Stop-Process -Id $_.ProcessId }
cd C:\Users\WorkHorizon\Desktop\html\next\data_fetcha5
python deploy\oracle\pack.py
```

This builds `deploy\oracle\paperbot.zip` with the bot, the trained model, and
your current balance and trades.

## 5. Copy it up and install

Replace `YOUR_IP` with the address from step 2:

```powershell
scp -i "$env:USERPROFILE\.ssh\paperbot.key" deploy\oracle\paperbot.zip ubuntu@YOUR_IP:~
ssh -i "$env:USERPROFILE\.ssh\paperbot.key" ubuntu@YOUR_IP
```

The first time, it asks whether to trust the server. Type `yes`. You are now
on the server. Run:

```bash
sudo apt-get install -y unzip && unzip paperbot.zip -d paperbot && bash paperbot/setup.sh
```

It checks that Binance is reachable, installs everything, runs the self-checks,
and starts the bot. It ends with `Done.`

## 6. Checking on it

Log in (the `ssh` line from step 5) and run:

```bash
~/status.sh
```

It shows the balance with and without fees, the number of trades, how many
were right, and any open trade.

Other useful commands on the server:

| What | Command |
|---|---|
| Watch it live (Ctrl+C to stop watching) | `journalctl -u paperbot -f` |
| Restart it | `sudo systemctl restart paperbot` |
| Stop it for good | `sudo systemctl disable --now paperbot` |
| Download the trade list to your PC (run on the PC) | `scp -i "$env:USERPROFILE\.ssh\paperbot.key" ubuntu@YOUR_IP:~/paperbot/backend/data/paper/BTCUSDT_trades.jsonl .` |

The bot restarts on its own after a crash or a server reboot, and it stops
opening trades when the 14 days end (2026-10-30 15:53 UTC).






Q7T7AxFpZAxqP4X