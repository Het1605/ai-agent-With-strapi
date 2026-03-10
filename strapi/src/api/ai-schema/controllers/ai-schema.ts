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

        if (!collectionName) {
            return ctx.badRequest('collectionName is required');
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

    async modifySchema(ctx: any) {
        const { collectionName, fields } = ctx.request.body;

        if (!collectionName || !Array.isArray(fields) || fields.length === 0) {
            return ctx.badRequest('collectionName and a non-empty fields array are required');
        }

        // Resolve the UID — same rule as createCollection
        const singularName = collectionName.toLowerCase().replace(/s$/, '');
        const uid = `api::${singularName}.${singularName}`;

        // @ts-ignore
        const existingContentType = strapi.contentTypes[uid];
        if (!existingContentType) {
            return ctx.badRequest(`Collection "${uid}" does not exist. Create it first.`);
        }

        // Gather existing attribute names to guard against overwrites
        const existingAttributes: any = existingContentType.attributes || {};
        const newAttributes: any = {};
        const skipped: string[] = [];
        const invalid: string[] = [];

        for (const field of fields) {
            if (!field.name || !field.type) {
                invalid.push(field.name || '(unnamed)');
                continue;
            }
            if (existingAttributes[field.name]) {
                skipped.push(field.name);
                continue;
            }
            newAttributes[field.name] = mapFieldToStrapiAttribute(field);
        }

        if (invalid.length > 0) {
            return ctx.badRequest(`Fields missing name or type: ${invalid.join(', ')}`);
        }

        if (Object.keys(newAttributes).length === 0) {
            return ctx.badRequest(
                skipped.length > 0
                    ? `All provided fields already exist: ${skipped.join(', ')}`
                    : 'No valid new fields to add'
            );
        }

        try {
            // @ts-ignore
            const contentTypeService = strapi.plugin('content-type-builder').service('content-types');

            // Merge new attributes on top of existing ones
            const mergedAttributes = { ...existingAttributes, ...newAttributes };

            await contentTypeService.editContentType(uid, {
                contentType: {
                    displayName: existingContentType.info?.displayName || singularName,
                    singularName: existingContentType.info?.singularName || singularName,
                    pluralName: existingContentType.info?.pluralName || `${singularName}s`,
                    kind: 'collectionType',
                    attributes: mergedAttributes,
                },
                components: [],
            });

            const response: any = {
                message: 'Schema updated successfully',
                collection: uid,
                added: Object.keys(newAttributes),
            };
            if (skipped.length > 0) response.skipped = skipped;

            ctx.send(response);
        } catch (error: any) {
            // @ts-ignore
            strapi.log.error('modifySchema error:', error);
            ctx.internalServerError(`Failed to modify schema: ${error.message}`);
        }
    },

    async fieldRegistry(ctx: any) {
        try {
            const fieldMap: Record<string, Set<string>> = {};

            // @ts-ignore
            const contentTypes = strapi.contentTypes;

            for (const uid in contentTypes) {
                const contentType = contentTypes[uid];
                const attributes = contentType.attributes;

                for (const attrName in attributes) {
                    const attr = attributes[attrName];
                    const type = attr.type;

                    if (!type) continue;

                    if (!fieldMap[type]) {
                        fieldMap[type] = new Set<string>();
                    }

                    // Dynamically collect all keys that are not 'type'
                    Object.keys(attr).forEach(key => {
                        if (key !== 'type') {
                            fieldMap[type].add(key);
                        }
                    });
                }
            }

            // Convert Sets to Arrays for JSON response
            const responseFields: Record<string, string[]> = {};
            for (const type in fieldMap) {
                responseFields[type] = Array.from(fieldMap[type]);
            }

            return ctx.send({
                fields: responseFields
            });
        } catch (error: any) {
            // @ts-ignore
            strapi.log.error('Field Registry Error:', error);
            return ctx.internalServerError(`Failed to retrieve field registry: ${error.message}`);
        }
    },
};
