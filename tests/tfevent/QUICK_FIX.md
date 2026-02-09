# QUICK FIX FOR PROTOBUF EXCEPTION ERROR

## Problem
```
exception (in /usr/lib/x86_64-linux-gnu/libprotobuf.so.17.0.0)
```

This means the proto files were compiled with a different protobuf version than the runtime library.

## Solution - Pick ONE:

---

## Option 1: Quick Fix Script (30 seconds)

```bash
chmod +x fix_protobuf.sh
./fix_protobuf.sh
```

Done! This automatically cleans and rebuilds everything with matching versions.

---

## Option 2: Manual Fix (1 minute)

```bash
# Clean old files
rm -f *.pb.cc *.pb.h tensorboard_writer

# Recompile protos
protoc --cpp_out=. histogram.proto
protoc --cpp_out=. summary.proto
protoc --cpp_out=. event.proto

# Rebuild app
g++ -std=c++14 -O2 \
    tensorboard_writer.cpp \
    event.pb.cc \
    summary.pb.cc \
    histogram.pb.cc \
    -lprotobuf \
    -o tensorboard_writer

# Run
./tensorboard_writer
```

---

## Option 3: Docker (100% Reliable - 2 minutes)

```bash
# Build and run
docker build -t tensorboard-writer .
docker run -v $(pwd)/logs:/app/logs tensorboard-writer

# View results
docker run -p 6006:6006 -v $(pwd)/logs:/app/logs tensorboard-writer \
    tensorboard --logdir=/app/logs --host=0.0.0.0
```

Or use docker-compose:
```bash
docker-compose up
# Then open http://localhost:6006
```

---

## Option 4: Using Makefile (1 minute)

```bash
make clean
make
make run
```

---

## Why Does This Happen?

Protocol Buffers requires that:
1. The version of `protoc` used to generate `.pb.cc` files
2. The version of `libprotobuf` linked at runtime
3. **MUST MATCH** (at least major.minor version)

Common mismatches:
- ✅ protoc 3.12.4 + libprotobuf 3.12.4 = Works
- ❌ protoc 3.12.4 + libprotobuf 3.6.1 = Fails
- ❌ protoc 3.21.0 + libprotobuf 3.12.4 = Fails

## Check Your Versions

```bash
# Check compiler version
protoc --version

# Check library version (Ubuntu/Debian)
dpkg -l | grep libprotobuf

# Check library version (Red Hat/CentOS)
rpm -qa | grep protobuf

# Check what your app is linked against
ldd tensorboard_writer | grep protobuf
```

## Still Not Working?

### If protoc is not installed:
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y protobuf-compiler libprotobuf-dev

# macOS
brew install protobuf

# Red Hat/CentOS
sudo yum install -y protobuf-devel protobuf-compiler
```

### If versions don't match:
```bash
# Reinstall both with matching versions
sudo apt-get remove --purge protobuf-compiler libprotobuf-dev
sudo apt-get autoremove
sudo apt-get install -y protobuf-compiler libprotobuf-dev

# Then rebuild
rm -f *.pb.cc *.pb.h
./build.sh
```

### If everything fails:
Use Docker (Option 3 above). It's guaranteed to work because everything runs in a controlled environment with matching versions.

## Prevention

Add to your workflow:

```bash
# Always clean before building
make clean
make

# Or
rm -f *.pb.cc *.pb.h
./build.sh
```

Add to `.gitignore`:
```
*.pb.cc
*.pb.h
tensorboard_writer
logs/
```

This prevents committing generated files that might have version mismatches.

## One-Line Nuclear Option

If you just want it to work and don't care about the details:

```bash
sudo apt-get remove --purge protobuf* && sudo apt-get install -y protobuf-compiler libprotobuf-dev && rm -f *.pb.* tensorboard_writer && ./build.sh
```

This:
1. Removes all protobuf packages
2. Reinstalls matching versions
3. Cleans generated files
4. Rebuilds everything

## Summary

**90% of the time:** Just run `./fix_protobuf.sh`

**If that doesn't work:** Use Docker

**For production:** Use Docker or pin specific protobuf versions in your CI/CD
