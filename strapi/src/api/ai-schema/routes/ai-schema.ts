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
    ],
};
