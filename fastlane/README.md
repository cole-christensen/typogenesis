fastlane documentation
----

# Installation

Make sure you have the latest version of the Xcode command line tools installed:

```sh
xcode-select --install
```

For _fastlane_ installation instructions, see [Installing _fastlane_](https://docs.fastlane.tools/#installing-fastlane)

# Available Actions

## Mac

### mac test

```sh
[bundle exec] fastlane mac test
```

Run all tests (unit + UI)

### mac unit_tests

```sh
[bundle exec] fastlane mac unit_tests
```

Run unit tests only

### mac ui_tests

```sh
[bundle exec] fastlane mac ui_tests
```

Run UI tests only

### mac build

```sh
[bundle exec] fastlane mac build
```

Build the app for testing

### mac build_developer_id

```sh
[bundle exec] fastlane mac build_developer_id
```

Build for Developer ID distribution (direct download)

### mac build_appstore

```sh
[bundle exec] fastlane mac build_appstore
```

Build for Mac App Store

### mac release_direct

```sh
[bundle exec] fastlane mac release_direct
```

Build, notarize, and prepare for direct distribution

### mac release_appstore

```sh
[bundle exec] fastlane mac release_appstore
```

Build and upload to Mac App Store

### mac screenshots

```sh
[bundle exec] fastlane mac screenshots
```

Capture screenshots of the app

### mac bump_build

```sh
[bundle exec] fastlane mac bump_build
```

Increment build number

### mac bump_version_patch

```sh
[bundle exec] fastlane mac bump_version_patch
```

Increment version number (patch)

### mac bump_version_minor

```sh
[bundle exec] fastlane mac bump_version_minor
```

Increment version number (minor)

### mac clean

```sh
[bundle exec] fastlane mac clean
```

Clean build artifacts

### mac regenerate_and_test

```sh
[bundle exec] fastlane mac regenerate_and_test
```

Generate xcodegen project and run tests

### mac ci

```sh
[bundle exec] fastlane mac ci
```

CI: Full test suite

### mac ci_pr

```sh
[bundle exec] fastlane mac ci_pr
```

CI: Build and test for PR

----

This README.md is auto-generated and will be re-generated every time [_fastlane_](https://fastlane.tools) is run.

More information about _fastlane_ can be found on [fastlane.tools](https://fastlane.tools).

The documentation of _fastlane_ can be found on [docs.fastlane.tools](https://docs.fastlane.tools).
