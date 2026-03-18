export default {
    routes: [
        {
            method: "POST",
            path: "/ai-schema/create-collection",
            handler: "ai-schema.createCollection",
            config: {
                auth: false,
                policies: [],
                middlewares: [],
            },
        },
        {
            method: "POST",
            path: "/ai-schema/modify-schema",
            handler: "ai-schema.modifySchema",
            config: {
                auth: false,
                policies: [],
                middlewares: [],
            },
        },
        {
            method: "GET",
            path: "/ai-schema/field-registry",
            handler: "ai-schema.fieldRegistry",
            config: {
                auth: false,
                policies: [],
                middlewares: [],
            },
        },
        {
            method: "GET",
            path: "/ai-schema/list-content-types",
            handler: "ai-schema.listContentTypes",
            config: {
                auth: false,
                policies: [],
                middlewares: [],
            },
        },
        {
            method: "GET",
            path: "/ai-schema/content-type-schema/:name",
            handler: "ai-schema.getContentTypeSchema",
            config: {
                auth: false,
                policies: [],
                middlewares: [],
            },
        },
        // ── DATA OPERATION ROUTES (Runtime CRUD) ──────────
        {
            method: "GET",
            path: "/ai-schema/get-entries/:collection",
            handler: "ai-schema.getEntries",
            config: {
                auth: false,
                policies: [],
                middlewares: [],
            },
        },
        {
            method: "GET",
            path: "/ai-schema/get-entry/:collection/:id",
            handler: "ai-schema.getEntry",
            config: {
                auth: false,
                policies: [],
                middlewares: [],
            },
        },
        {
            method: "POST",
            path: "/ai-schema/create-entry",
            handler: "ai-schema.createEntry",
            config: {
                auth: false,
                policies: [],
                middlewares: [],
            },
        },
        {
            method: "PUT",
            path: "/ai-schema/update-entry",
            handler: "ai-schema.updateEntry",
            config: {
                auth: false,
                policies: [],
                middlewares: [],
            },
        },
        {
            method: "DELETE",
            path: "/ai-schema/delete-entry/:collection/:id",
            handler: "ai-schema.deleteEntry",
            config: {
                auth: false,
                policies: [],
                middlewares: [],
            },
        },
    ],
};
