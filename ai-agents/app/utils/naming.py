def to_strapi_slug(name: str) -> str:
    """
    Converts a raw table or field name to a Strapi-compatible slug (lower-hyphenated).
    Ex: 'Employee Leave' -> 'employee-leave'
    Ex: 'employee_leave' -> 'employee-leave'
    """
    if not name:
        return "untitled"
    return name.lower().replace(" ", "_").replace("_", "-")

def singularize(name: str) -> str:
    """
    Basic singularization logic that avoids mangling 'ss' endings (like address).
    Ex: 'customers' -> 'customer'
    Ex: 'address' -> 'address'
    """
    if not name:
        return name
    if name.endswith('s') and not name.endswith('ss'):
        return name[:-1]
    return name

def get_strapi_uid(table_name: str) -> str:
    """
    Generates a full Strapi UID from a raw table name.
    Ex: 'Customers' -> 'api::customer.customer'
    """
    slug = to_strapi_slug(table_name)
    singular = singularize(slug)
    return f"api::{singular}.{singular}"
