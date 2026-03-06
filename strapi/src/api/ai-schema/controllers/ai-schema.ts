export default {
    async createCollection(ctx: any) {
        const { collectionName, fields } = ctx.request.body;

        if (!collectionName || !fields) {
            return ctx.badRequest('collectionName and fields are required');
        }

        const singularName = collectionName.toLowerCase().replace(/s$/, ''); // Basic singularization
        const pluralName = collectionName.toLowerCase();
        const displayName = collectionName.charAt(0).toUpperCase() + collectionName.slice(1);
        const uid = `api::${singularName}.${singularName}`;

        // Check if it already exists to avoid 500 errors
        // @ts-ignore
        if (strapi.contentTypes[uid]) {
            return ctx.badRequest(`Collection "${uid}" already exists.`);
        }

        const attributes: any = {};

        for (const field of fields) {
            let type = field.type;

            // Strapi v5 prefers granular types instead of generic 'number'
            if (type === 'number') {
                type = field.numberSize || 'integer';
            }

            const fieldConfig: any = {
                type: type,
            };

            if (field.required !== undefined) fieldConfig.required = field.required;
            if (field.unique !== undefined) fieldConfig.unique = field.unique;
            if (field.default !== undefined) fieldConfig.default = field.default;

            // Type-specific mappings
            if (type === 'enumeration') {
                fieldConfig.enum = field.enum || [];
            } else if (type === 'relation') {
                fieldConfig.relation = field.relation;
                fieldConfig.target = field.target;
                if (field.targetAttribute) fieldConfig.targetAttribute = field.targetAttribute;
            } else if (type === 'uid') {
                fieldConfig.targetField = field.targetField;
            } else if (type === 'media') {
                fieldConfig.multiple = field.multiple || false;
                fieldConfig.allowedTypes = field.allowedTypes || ['images', 'files', 'videos', 'audios'];
            } else if (type === 'blocks') {
                // Rich text (Blocks) - The new JSON-based rich text editor
            } else if (type === 'richtext') {
                // Rich text (Markdown) - The classic rich text editor
            } else if (type === 'json') {
                // JSON - Data in JSON format
            } else if (type === 'password') {
                // Password - Password field with encryption
            } else if (type === 'email') {
                // Email - Email field with validation format
            } else if (type === 'boolean') {
                // Boolean - Yes or no, 1 or 0, true or false
            } else if (type === 'date' || type === 'datetime' || type === 'time' || type === 'timestamp') {
                // Date/Time pickers
            }

            attributes[field.name] = fieldConfig;
        }

        try {
            // @ts-ignore
            const contentTypeService = strapi.plugin('content-type-builder').service('content-types');

            // Strapi v5 createContentType expects an object with contentType and components
            const result = await contentTypeService.createContentType({
                contentType: {
                    displayName,
                    singularName,
                    pluralName,
                    kind: 'collectionType',
                    attributes,
                },
                components: [],
            });

            // The service call triggers a hot-reload in development mode.
            // We respond immediately to ensure the client gets a success message
            // before the server process might restart.
            ctx.send({
                message: 'Collection created successfully',
                uid: result.uid,
            });
        } catch (error: any) {
            // @ts-ignore
            strapi.log.error(error);

            if (error.message === 'contentType.alreadyExists') {
                return ctx.badRequest('Collection already exists');
            }

            ctx.internalServerError(`Failed to create collection: ${error.message}`);
        }
    },
};
