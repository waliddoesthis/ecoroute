# Shared EcoRoute API on an Oracle Cloud "Always Free" VM

At no cost, this gives a server that stays on, with HTTPS, running the gateway in hosted
mode: every caller pays with their own provider key, and only holders of an access key
get in.

## 1. Create the VM (Oracle Cloud console, about 10 minutes)

1. Sign up at cloud.oracle.com. A card is asked for to verify your identity. Always Free
   resources are not charged (check Oracle's current terms when you sign up).
2. **Compute → Instances → Create instance.**
   - Image: **Canonical Ubuntu 24.04** (or 22.04).
   - Shape: **Change shape → Ampere → VM.Standard.A1.Flex**, 2 OCPUs and 12 GB of memory
     (it is marked "Always Free-eligible"; up to 4 OCPUs and 24 GB is free in total).
   - Networking: keep "Assign a public IPv4 address" on.
   - SSH keys: download the generated private key, or paste your own public key.
   - If it says "Out of capacity", pick another availability domain, or try again later.
3. **Open ports 80 and 443:** open the instance's subnet, then its **Security List → Add
   Ingress Rules**: source `0.0.0.0/0`, TCP, destination ports `80,443`.

## 2. Install EcoRoute (one command)

```bash
ssh -i <your-key> ubuntu@<public-ip>
curl -fsSL https://raw.githubusercontent.com/waliddoesthis/ecoroute/main/deploy/vm/setup.sh | sudo bash
```

The first build takes 10 to 20 minutes. At the end it prints:

- the API address, `https://<ip-with-dashes>.sslip.io`. A free certificate comes from
  Let's Encrypt, and no domain is needed (`ECOROUTE_DOMAIN=api.example.com` uses yours);
- a generated access key, stored in `/opt/ecoroute/deploy/vm/.env`. Add more keys there,
  comma separated, then run the script again;
- the one-line command each person runs to connect Claude Code.

Running the script again updates EcoRoute to the latest `main`. The gateway and Caddy
restart by themselves after a reboot.

## Check it

```bash
curl https://<address>/health                 # {"status":"ok",...}
curl -X POST https://<address>/route -H "X-EcoRoute-Key: <key>" \
  -H "Content-Type: application/json" -d '{"messages":[{"role":"user","content":"hi"}]}'
```
