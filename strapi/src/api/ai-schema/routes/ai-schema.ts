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
    ],
};
