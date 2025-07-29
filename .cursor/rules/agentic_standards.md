# Standards for Cursor Code Generation

- The codebase should not have deprecated functions. Every function should be used and should migrate all functionality to use new functions.
- We should use DTO's and VO's instead of Dicts
- When a new function is created, unit tests should be made.
- When running any commands, make sure we are in the venv
- Any new libraries added should be added to requirements.txt and are comapatible
- Always make sure that all of the imports added are used