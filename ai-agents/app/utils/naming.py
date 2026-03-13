def to_strapi_slug(name: str) -> str:
    """
    Converts a raw table or field name to a Strapi-compatible slug (lower-hyphenated).
    Ex: 'Employee Leave' -> 'employee-leave'
    Ex: 'employee_leave' -> 'employee-leave'
    """
    if not name:
        return "untitled"
    # Strapi content type IDs use lower-kebab-case
    return name.lower().strip().replace(" ", "-").replace("_", "-")

def get_strapi_uid(singular_name: str) -> str:
    """
    Generates a full Strapi UID from a singular name.
    Ex: 'customer' -> 'api::customer.customer'
    """
    return f"api::{singular_name}.{singular_name}"
