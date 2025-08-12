# Data Transfer Object (DTO) Rules

## Core Principles

### 1. **Single Responsibility**
- DTOs should have one clear purpose: transferring data between layers
- Avoid mixing business logic with data transfer concerns
- Keep DTOs focused on data representation only

### 2. **Immutability**
- Make DTOs immutable when possible (use `readonly` properties in TypeScript/C#)
- Prefer constructor-based initialization over property setters
- This prevents accidental modifications during data transfer

### 3. **Validation**
- Validate data at the boundaries (API endpoints, service layer entry points)
- Use validation attributes/decorators for input DTOs
- Return validation errors as part of the response, not exceptions

## Naming Conventions

### 4. **Consistent Naming**
- Use descriptive, domain-specific names
- Follow language conventions:
  - **C#/Java**: `UserDto`, `CreateUserRequest`, `UserResponse`
  - **TypeScript**: `UserDTO`, `CreateUserRequest`, `UserResponse`
  - **Python**: `UserDTO`, `CreateUserRequest`, `UserResponse`
- Suffix with purpose: `Request`, `Response`, `Command`, `Query`

### 5. **Clear Intent**
- Request DTOs: `CreateUserRequest`, `UpdateUserRequest`
- Response DTOs: `UserResponse`, `UserListResponse`
- Query DTOs: `UserSearchQuery`, `UserFilterRequest`
- Command DTOs: `CreateUserCommand`, `DeleteUserCommand`

## Structure Guidelines

### 6. **Flat Structure**
- Prefer flat structures over deeply nested objects
- Avoid circular references
- Use composition over inheritance for DTOs

### 7. **Property Naming**
- Use clear, descriptive property names
- Follow camelCase for JSON serialization
- Use consistent naming across related DTOs

### 8. **Data Types**
- Use primitive types when possible
- Avoid complex objects in DTOs unless necessary
- Use enums for fixed sets of values
- Use nullable types appropriately

## Best Practices

### 9. **Separation of Concerns**
- Keep DTOs separate from domain entities
- Don't expose internal implementation details
- Use mapping libraries (AutoMapper, MapStruct) for conversions

### 10. **Versioning**
- Include version information in DTOs when needed
- Use semantic versioning for API changes
- Consider backward compatibility

### 11. **Documentation**
- Document all DTO properties with clear descriptions
- Include examples for complex DTOs

### 12. **Error Handling**
- Include error information in response DTOs
- Use consistent error response structures
- Provide meaningful error messages

## Common Patterns

### 13. **Request/Response Pattern**
```typescript
// Request DTO
interface CreateUserRequest {
  readonly email: string;
  readonly name: string;
  readonly age?: number;
}

// Response DTO
interface UserResponse {
  readonly id: string;
  readonly email: string;
  readonly name: string;
  readonly age?: number;
  readonly createdAt: Date;
}
```

### 14. **Pagination Pattern**
```typescript
interface PaginatedResponse<T> {
  readonly data: T[];
  readonly totalCount: number;
  readonly pageNumber: number;
  readonly pageSize: number;
  readonly totalPages: number;
}
```

### 15. **Filter/Search Pattern**
```typescript
interface UserSearchRequest {
  readonly query?: string;
  readonly filters?: {
    readonly ageRange?: { min: number; max: number };
    readonly status?: UserStatus[];
  };
  readonly sortBy?: string;
  readonly sortOrder?: 'asc' | 'desc';
  readonly pageNumber: number;
  readonly pageSize: number;
}
```

## Anti-Patterns to Avoid

### 16. **Don't Do This**
- ❌ Mixing business logic in DTOs
- ❌ Using DTOs as domain entities
- ❌ Exposing internal database fields
- ❌ Creating overly complex nested structures
- ❌ Using DTOs for internal communication
- ❌ Ignoring validation at boundaries

### 17. **Security Considerations**
- Never expose sensitive data in DTOs
- Sanitize input data
- Use DTOs to control what data is exposed
- Validate all inputs thoroughly

## Language-Specific Guidelines

### 18. **TypeScript/JavaScript**
```typescript
// Use interfaces for DTOs
interface UserDTO {
  readonly id: string;
  readonly email: string;
  readonly name: string;
}

// Use type unions for responses
type ApiResponse<T> = 
  | { success: true; data: T }
  | { success: false; error: string };
```

### 19. **C#**
```csharp
public record CreateUserRequest(
    string Email,
    string Name,
    int? Age = null
);

public record UserResponse(
    string Id,
    string Email,
    string Name,
    int? Age,
    DateTime CreatedAt
);
```

### 20. **Python**
```python
from dataclasses import dataclass
from typing import Optional
from datetime import datetime

@dataclass(frozen=True)
class CreateUserRequest:
    email: str
    name: str
    age: Optional[int] = None

@dataclass(frozen=True)
class UserResponse:
    id: str
    email: str
    name: str
    age: Optional[int]
    created_at: datetime
```

## Testing Guidelines

### 21. **DTO Testing**
- Test DTO serialization/deserialization
- Validate mapping between DTOs and domain objects
- Test edge cases and boundary conditions
- Use factories for creating test DTOs

### 22. **Validation Testing**
- Test all validation rules
- Test both valid and invalid inputs
- Test boundary conditions
- Mock external dependencies

## Performance Considerations

### 23. **Optimization**
- Keep DTOs lightweight
- Avoid unnecessary properties
- Use lazy loading for large datasets
- Consider using projection queries for large objects

### 24. **Caching**
- DTOs should be cacheable when appropriate
- Use appropriate cache keys
- Consider cache invalidation strategies
- Document caching behavior
