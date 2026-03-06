const mapFieldToStrapiAttribute = (field: any) => {
    // 1. Determine base type (Handle Strapi v5 number granularity)
    let type = field.type;
    if (type === 'number') {
        type = field.numberSize || 'integer';
    }

    // 2. Initialize base config
    const config: any = { type };

    // 3. Map Global Options (Present in most types)
    const globalOptions = ['required', 'unique', 'default', 'private', 'configurable'];
    globalOptions.forEach(opt => {
        if (field[opt] !== undefined) config[opt] = field[opt];
    });

    // 4. Map Type-Specific Validations/Options
    switch (type) {
        case 'string':
        case 'text':
        case 'richtext':
        case 'email':
        case 'password':
        case 'uid':
            if (field.minLength !== undefined) config.minLength = field.minLength;
            if (field.maxLength !== undefined) config.maxLength = field.maxLength;
            if (field.regex !== undefined) config.regex = field.regex;
            if (type === 'uid' && field.targetField) config.targetField = field.targetField;
            break;

        case 'integer':
        case 'biginteger':
        case 'decimal':
        case 'float':
            if (field.min !== undefined) config.min = field.min;
            if (field.max !== undefined) config.max = field.max;
            break;

        case 'enumeration':
            config.enum = field.enum || [];
            break;

        case 'media':
            config.multiple = !!field.multiple;
            config.allowedTypes = field.allowedTypes || ['images', 'files', 'videos', 'audios'];
            break;

        case 'relation':
            config.relation = field.relation;
            config.target = field.target;
            if (field.targetAttribute) config.targetAttribute = field.targetAttribute;
            break;

        default:
            // Other types (blocks, json, boolean, date) usually only use global options
            break;
    }

    return config;
};

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
            attributes[field.name] = mapFieldToStrapiAttribute(field);
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
